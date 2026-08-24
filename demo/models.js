// Extend the compact canvas viewer with model selection and a prompt sequence.
window.addEventListener('load', async () => {
  const prompt = document.querySelector('#prompt');
  const form = prompt?.closest('.promptbox');
  const generate = document.querySelector('#generate');
  if (!prompt || !form || !generate) return;
  const models = await fetch('/api/models').then(r => r.json());

  const modelLabel = document.createElement('label');
  modelLabel.htmlFor = 'motionModel'; modelLabel.textContent = 'Motion model';
  const select = document.createElement('select');
  select.id = 'motionModel'; select.style.cssText = 'width:100%;padding:10px;border-radius:10px;background:#0b1015;color:#f3f6f4;border:1px solid #25313b';
  for (const model of models) {
    const option = document.createElement('option'); option.value = model.id;
    option.disabled = !model.available;
    option.textContent = `${model.label}${model.available ? '' : ' — coming soon'}`;
    select.append(option);
  }
  const modelHint = document.createElement('div'); modelHint.className = 'hint';
  const updateModel = () => {
    const model = models.find(item => item.id === select.value);
    modelHint.textContent = model.available ? `${model.skeleton} · ${model.upstream}` : `${model.skeleton} · ${model.reason}`;
  };
  select.onchange = updateModel;
  form.insertBefore(modelLabel, prompt); form.insertBefore(select, prompt); form.insertBefore(modelHint, prompt); updateModel();

  const sequence = document.createElement('div');
  sequence.style.cssText = 'display:grid;gap:10px;width:100%';
  prompt.before(sequence);
  prompt.classList.add('sequence-prompt');
  const segmentControls = new Map();
  const primaryRow = document.createElement('div');
  primaryRow.style.cssText = 'display:grid;grid-template-columns:1fr 74px;gap:7px;align-items:start';
  const primaryDuration = document.createElement('input');
  primaryDuration.type = 'number'; primaryDuration.min = '60'; primaryDuration.max = '300'; primaryDuration.step = '30'; primaryDuration.value = '150';
  primaryDuration.title = 'Frames (60–300)';
  primaryRow.append(prompt, primaryDuration); sequence.append(primaryRow);
  segmentControls.set(primaryRow, primaryDuration);
  const count = document.createElement('div'); count.className = 'hint';
  const updateCount = () => {
    const prompts = sequence.querySelectorAll('.sequence-prompt');
    count.textContent = `${prompts.length} segment${prompts.length === 1 ? '' : 's'} · 5-frame overlap`;
  };
  const addSegment = (text = '', frames = 150) => {
    const row = document.createElement('div'); row.style.cssText = 'display:grid;grid-template-columns:1fr 74px auto;gap:7px;align-items:start';
    const textArea = document.createElement('textarea'); textArea.className = 'sequence-prompt'; textArea.value = text;
    textArea.placeholder = 'Describe the next motion'; textArea.style.minHeight = '64px';
    const duration = document.createElement('input'); duration.type = 'number'; duration.min = '60'; duration.max = '300'; duration.step = '30'; duration.value = String(frames); duration.title = 'Frames (60–300)';
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '×'; remove.title = 'Remove segment'; remove.style.cssText = 'padding:8px 12px;background:#24313a;color:#dce9e8';
    remove.onclick = () => { row.remove(); updateCount(); };
    row.append(textArea, duration, remove); sequence.append(row); segmentControls.set(row, duration);
    updateCount();
  };
  const add = document.createElement('button'); add.type = 'button'; add.textContent = '+ Add prompt segment';
  add.style.cssText = 'justify-self:start;padding:8px 12px;background:#24313a;color:#dce9e8';
  add.onclick = () => addSegment(); form.insertBefore(add, generate); form.insertBefore(count, generate); updateCount();

  // The gallery owns the selected animation; receive its full saved sequence
  // rather than restoring only animation.prompt (the first segment).
  window.addEventListener('kimodo:restore-sequence', event => {
    const {segments, model} = event.detail || {};
    if (model && [...select.options].some(option => option.value === model)) {
      select.value = model;
      updateModel();
    }
    const restored = Array.isArray(segments) && segments.length
      ? segments
      : [{prompt: prompt.value, frames: 150}];
    const first = restored[0];
    prompt.value = first.prompt || '';
    primaryDuration.value = String(first.frames || 150);
    for (const row of [...sequence.children]) {
      if (row !== primaryRow) row.remove();
    }
    for (const segment of restored.slice(1)) {
      addSegment(segment.prompt || '', segment.frames || 150);
    }
    updateCount();
  });

  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    if (typeof input === 'string' && input.endsWith('/api/generate') && init?.body) {
      const body = JSON.parse(init.body);
      body.model = select.value;
      body.transition_frames = 5;
      body.segments = [...sequence.querySelectorAll('.sequence-prompt')].map(area => {
        const row = area.closest('div');
        const duration = segmentControls.get(row);
        return {prompt: area.value, frames: Number(duration?.value || 150)};
      });
      return nativeFetch(input, {...init, body: JSON.stringify(body)});
    }
    return nativeFetch(input, init);
  };
});
