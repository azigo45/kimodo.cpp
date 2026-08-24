package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStitchSegmentsBlendsOverlap(t *testing.T) {
	dir := t.TempDir()
	first, second := filepath.Join(dir, "first"), filepath.Join(dir, "second")
	if err := os.MkdirAll(first, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(second, 0755); err != nil {
		t.Fatal(err)
	}
	rot := make([]float32, 2*22*4)
	for index := range rot {
		if index%4 == 3 {
			rot[index] = 1
		}
	}
	if err := writeF32(filepath.Join(first, "root_positions.f32"), []float32{0, 0, 0, 1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := writeF32(filepath.Join(first, "local_rotations_xyzw.f32"), rot); err != nil {
		t.Fatal(err)
	}
	if err := writeF32(filepath.Join(second, "root_positions.f32"), []float32{3, 0, 0, 5, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := writeF32(filepath.Join(second, "local_rotations_xyzw.f32"), rot); err != nil {
		t.Fatal(err)
	}
	if err := stitchSegments(dir, []string{first, second}, 1); err != nil {
		t.Fatal(err)
	}
	root, err := readF32(filepath.Join(dir, "root_positions.f32"))
	if err != nil {
		t.Fatal(err)
	}
	if len(root) != 9 {
		t.Fatalf("frames = %d, want 3", len(root)/3)
	}
	if root[3] != 2 {
		t.Fatalf("blended boundary = %v, want 2", root[3])
	}
}
