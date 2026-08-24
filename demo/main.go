// Local Kimodo text-to-motion demo. Generation is serialized so one native
// process owns Vulkan at a time, while the persistent gallery stays readable.
package main

import (
	"crypto/rand"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed index.html
var files embed.FS

//go:embed models.js
var modelUI []byte

//go:embed assets/localai.png
var localAILogo []byte

type animation struct {
	ID               string          `json:"id"`
	Prompt           string          `json:"prompt"`
	Frames           int             `json:"frames"`
	DiffusionSteps   int             `json:"diffusion_steps"`
	Seed             uint64          `json:"seed"`
	CreatedAt        string          `json:"created_at"`
	Status           string          `json:"status"`
	Error            string          `json:"error,omitempty"`
	Kind             string          `json:"kind"`
	Model            string          `json:"model"`
	Segments         []promptSegment `json:"segments,omitempty"`
	TransitionFrames int             `json:"transition_frames,omitempty"`
	Progress         string          `json:"progress,omitempty"`
}
type promptSegment struct {
	Prompt string `json:"prompt"`
	Frames int    `json:"frames"`
}
type motionModel struct {
	ID        string `json:"id"`
	Label     string `json:"label"`
	Skeleton  string `json:"skeleton"`
	Upstream  string `json:"upstream"`
	Available bool   `json:"available"`
	Reason    string `json:"reason,omitempty"`
	Motion    string `json:"-"`
}
type gallery struct {
	mu                      sync.RWMutex
	items                   map[string]*animation
	output                  string
	queue                   chan string
	generator, motion, text string
	models                  map[string]motionModel
}

func token() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}
func (g *gallery) save(a *animation) error {
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(g.output, a.ID+".json"), b, 0644)
}
func (g *gallery) list() []*animation {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make([]*animation, 0, len(g.items))
	for _, item := range g.items {
		copy := *item
		result = append(result, &copy)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].CreatedAt > result[j].CreatedAt })
	return result
}

func copyFile(dst, src string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0600)
}

func readF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("invalid F32 file: %s", path)
	}
	values := make([]float32, len(b)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return values, nil
}

func writeF32(path string, values []float32) error {
	b := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(value))
	}
	return os.WriteFile(path, b, 0600)
}

func blendQuaternion(a, b []float32, alpha float32) {
	dot := a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
	if dot < 0 {
		for i := range b {
			b[i] = -b[i]
		}
	}
	length := float32(0)
	for i := range a {
		a[i] = alpha*a[i] + (1-alpha)*b[i]
		length += a[i] * a[i]
	}
	if length > 0 {
		length = 1 / float32(math.Sqrt(float64(length)))
		for i := range a {
			a[i] *= length
		}
	}
}

// stitchSegments joins independently sampled demo segments. The overlap is
// blended in root space and by normalized linear interpolation for quaternions.
// Native observed-motion conditioning is deliberately a later parity step.
func stitchSegments(output string, dirs []string, overlap int) error {
	var roots, rotations []float32
	for index, dir := range dirs {
		root, err := readF32(filepath.Join(dir, "root_positions.f32"))
		if err != nil {
			return err
		}
		rot, err := readF32(filepath.Join(dir, "local_rotations_xyzw.f32"))
		if err != nil {
			return err
		}
		frames := len(root) / 3
		if frames == 0 || len(rot) != frames*22*4 {
			return fmt.Errorf("invalid motion segment %d", index+1)
		}
		if index == 0 {
			roots, rotations = root, rot
			continue
		}
		n := overlap
		if n > frames {
			n = frames
		}
		if n > len(roots)/3 {
			n = len(roots) / 3
		}
		for frame := 0; frame < n; frame++ {
			alpha := float32(0.5)
			if n > 1 {
				alpha = 1 - float32(frame)/float32(n-1)
			}
			old := (len(roots)/3 - n + frame) * 3
			newest := frame * 3
			for axis := 0; axis < 3; axis++ {
				roots[old+axis] = alpha*roots[old+axis] + (1-alpha)*root[newest+axis]
			}
			for joint := 0; joint < 22; joint++ {
				oldQ := (len(rotations)/4 - n*22 + frame*22 + joint) * 4
				newQ := (frame*22 + joint) * 4
				blendQuaternion(rotations[oldQ:oldQ+4], append([]float32(nil), rot[newQ:newQ+4]...), alpha)
			}
		}
		roots = append(roots, root[n*3:]...)
		rotations = append(rotations, rot[n*22*4:]...)
	}
	if err := writeF32(filepath.Join(output, "root_positions.f32"), roots); err != nil {
		return err
	}
	return writeF32(filepath.Join(output, "local_rotations_xyzw.f32"), rotations)
}
func (g *gallery) worker() {
	for id := range g.queue {
		g.mu.Lock()
		item := g.items[id]
		item.Status = "running"
		_ = g.save(item)
		g.mu.Unlock()
		dir := filepath.Join(g.output, id)
		err := os.MkdirAll(dir, 0755)
		if err == nil {
			err = os.WriteFile(filepath.Join(dir, "prompt.txt"), []byte(item.Prompt), 0600)
		}
		if err == nil {
			model, ok := g.models[item.Model]
			if !ok || !model.Available {
				err = fmt.Errorf("model %q is not available", item.Model)
			} else {
				segments := item.Segments
				if len(segments) == 0 {
					segments = []promptSegment{{Prompt: item.Prompt, Frames: item.Frames}}
				}
				args := []string{model.Motion, g.text, "--sequence", fmt.Sprint(item.TransitionFrames), fmt.Sprint(item.DiffusionSteps), fmt.Sprint(item.Seed), dir}
				for index, segment := range segments {
					promptPath := filepath.Join(dir, fmt.Sprintf("segment-%02d.txt", index+1))
					if err = os.WriteFile(promptPath, []byte(segment.Prompt), 0600); err != nil {
						break
					}
					args = append(args, fmt.Sprint(segment.Frames), promptPath)
				}
				if err == nil {
					g.mu.Lock()
					item.Progress = fmt.Sprintf("Generating %d conditioned segments", len(segments))
					_ = g.save(item)
					g.mu.Unlock()
					cmd := exec.Command(g.generator, args...)
					cmd.Env = append(os.Environ(), "KIMODO_BACKEND=vulkan")
					output, runErr := cmd.CombinedOutput()
					if runErr != nil {
						err = fmt.Errorf("sequence: %w: %s", runErr, strings.TrimSpace(string(output)))
					}
				}
			}
		}
		g.mu.Lock()
		if err != nil {
			item.Status = "failed"
			item.Error = err.Error()
		} else {
			item.Status = "ready"
			item.Progress = ""
		}
		if saveErr := g.save(item); saveErr != nil {
			log.Printf("save %s: %v", item.ID, saveErr)
		}
		g.mu.Unlock()
	}
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8090", "listen address")
	motion := flag.String("motion-model", "models/kimodo-smplx-rp-v1-f32.gguf", "motion GGUF")
	text := flag.String("text-bundle", "generated/llm2vec-text-bundle", "native LLM2Vec component directory")
	generator := flag.String("generator", "build/debug/kmd-generate", "native text-to-motion command")
	output := flag.String("output", "demo-output", "persistent gallery directory")
	flag.Parse()
	if err := os.MkdirAll(*output, 0755); err != nil {
		log.Fatal(err)
	}
	models := map[string]motionModel{
		"smplx-rp-v1":    {ID: "smplx-rp-v1", Label: "SMPL-X RP v1", Skeleton: "SMPL-X 22 joints", Upstream: "nvidia/Kimodo-SMPLX-RP-v1", Available: true, Motion: *motion},
		"soma-rp-v1.1":   {ID: "soma-rp-v1.1", Label: "SOMA RP v1.1", Skeleton: "SOMA 30 joints", Upstream: "nvidia/Kimodo-SOMA-RP-v1.1", Reason: "SOMA decoder and GGML conversion are being added"},
		"soma-seed-v1.1": {ID: "soma-seed-v1.1", Label: "SOMA SEED v1.1", Skeleton: "SOMA 30 joints", Upstream: "nvidia/Kimodo-SOMA-SEED-v1.1", Reason: "SOMA decoder and GGML conversion are being added"},
		"g1-rp-v1":       {ID: "g1-rp-v1", Label: "G1 RP v1", Skeleton: "Unitree G1 34 joints", Upstream: "nvidia/Kimodo-G1-RP-v1", Reason: "G1 decoder and GGML conversion are being added"},
		"g1-seed-v1":     {ID: "g1-seed-v1", Label: "G1 SEED v1", Skeleton: "Unitree G1 34 joints", Upstream: "nvidia/Kimodo-G1-SEED-v1", Reason: "G1 decoder and GGML conversion are being added"},
	}
	g := &gallery{items: map[string]*animation{}, output: *output, queue: make(chan string, 32), generator: *generator, motion: *motion, text: *text, models: models}
	entries, _ := filepath.Glob(filepath.Join(*output, "*.json"))
	for _, path := range entries {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var a animation
		if json.Unmarshal(b, &a) == nil {
			g.items[a.ID] = &a
		}
	}
	go g.worker()
	index, err := files.ReadFile("index.html")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(index)
	})
	mux.HandleFunc("/localai.png", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		_, _ = w.Write(localAILogo)
	})
	mux.HandleFunc("/models.js", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(modelUI)
	})
	mux.HandleFunc("/api/animations", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(g.list())
	})
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		result := make([]motionModel, 0, len(g.models))
		for _, model := range g.models {
			result = append(result, model)
		}
		sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(result)
	})
	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		var request struct {
			Prompt           string          `json:"prompt"`
			Segments         []promptSegment `json:"segments"`
			TransitionFrames int             `json:"transition_frames"`
			Frames           int             `json:"frames"`
			Steps            int             `json:"steps"`
			Seed             uint64          `json:"seed"`
			Model            string          `json:"model"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<10)).Decode(&request); err != nil {
			http.Error(w, "invalid JSON", 400)
			return
		}
		request.Prompt = strings.TrimSpace(request.Prompt)
		if len(request.Segments) == 0 {
			request.Segments = []promptSegment{{Prompt: request.Prompt, Frames: request.Frames}}
		}
		if len(request.Segments) > 16 {
			http.Error(w, "at most 16 prompt segments", 400)
			return
		}
		if request.Frames == 0 {
			request.Frames = 150
		}
		if request.Steps == 0 {
			request.Steps = 100
		}
		for index := range request.Segments {
			request.Segments[index].Prompt = strings.TrimSpace(request.Segments[index].Prompt)
			if request.Segments[index].Frames == 0 {
				request.Segments[index].Frames = 150
			}
			if request.Segments[index].Prompt == "" || len(request.Segments[index].Prompt) > 4096 || request.Segments[index].Frames < 60 || request.Segments[index].Frames > 300 {
				http.Error(w, "each prompt segment must be 60..300 frames and 1..4096 bytes", 400)
				return
			}
		}
		if request.TransitionFrames == 0 {
			request.TransitionFrames = 5
		}
		if request.TransitionFrames < 1 || request.TransitionFrames > 60 || request.Steps < 1 || request.Steps > 1000 {
			http.Error(w, "transition frames must be 1..60 and steps 1..1000", 400)
			return
		}
		if request.Model == "" {
			request.Model = "smplx-rp-v1"
		}
		model, ok := g.models[request.Model]
		if !ok || !model.Available {
			http.Error(w, "selected motion model is not available: "+model.Reason, http.StatusConflict)
			return
		}
		totalFrames := 0
		for _, segment := range request.Segments {
			totalFrames += segment.Frames
		}
		totalFrames -= request.TransitionFrames * (len(request.Segments) - 1)
		a := &animation{ID: token(), Prompt: request.Segments[0].Prompt, Frames: totalFrames, DiffusionSteps: request.Steps, Seed: request.Seed, CreatedAt: time.Now().UTC().Format(time.RFC3339), Status: "queued", Kind: "generated", Model: request.Model, Segments: request.Segments, TransitionFrames: request.TransitionFrames}
		g.mu.Lock()
		g.items[a.ID] = a
		err := g.save(a)
		g.mu.Unlock()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		g.queue <- a.ID
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(a)
	})
	mux.HandleFunc("/api/animations/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/animations/"), "/")
		if len(parts) != 2 || (parts[1] != "root.f32" && parts[1] != "rotations.f32") {
			http.NotFound(w, r)
			return
		}
		g.mu.RLock()
		a := g.items[parts[0]]
		g.mu.RUnlock()
		if a == nil || a.Status != "ready" {
			http.NotFound(w, r)
			return
		}
		name := "root_positions.f32"
		if parts[1] == "rotations.f32" {
			name = "local_rotations_xyzw.f32"
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Cache-Control", "no-store")
		http.ServeFile(w, r, filepath.Join(g.output, a.ID, name))
	})
	log.Printf("Kimodo text-to-motion demo listening at http://%s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
