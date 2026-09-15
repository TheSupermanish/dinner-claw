let state = null, arm = 'left', connected = false, pending = 0;
let frameURL = null, queue = Promise.resolve(), lastFrame = 0, freshFrame = true;
const $ = id => document.getElementById(id);

function message(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}

async function request(path, body) {
  let response;
  try {
    response = await fetch('/api/' + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? {} : {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(8000), cache: 'no-store'
    });
  } catch (error) {
    error.offline = true;
    throw error;
  }
  if (!response.ok) {
    const result = await response.json();
    const error = Error(result.error || 'Simulation request failed');
    error.offline = response.status >= 500;
    throw error;
  }
  return response;
}
async function api(path, body) { return (await request(path, body)).json(); }

function availability(value) {
  connected = value;
  $('connection').classList.toggle('offline', !value);
  $('viewport').classList.toggle('disconnected', !value);
  $('retry').hidden = value;
  for (const id of ['pickup', 'drawer', 'fork', 'plate', 'bimanual', 'execute', 'learned', 'hybrid', 'demo', 'play', 'step', 'reach', 'reset']) $(id).disabled = !value;
  for (const input of document.querySelectorAll('#joints input')) input.disabled = !value;
  if (!value) {
    $('connection').textContent = '● Disconnected — retrying';
    $('runstatus').textContent = 'Live simulation unavailable. The image is the last received frame. Reconnecting automatically…';
  }
}

function display(s) {
  state = s;
  for (const id of ['learned', 'hybrid']) $(id).disabled = !connected || !s.capabilities.learned_policy;
  $('learnedstatus').textContent = s.capabilities.learned_policy
    ? 'FP32 ACT checkpoint available · Experimental · Mac CPU, not qualifying Intel hardware.'
    : 'Not installed. Train and export the ACT checkpoint and install the Intel extra first.';
  $('time').textContent = s.time.toFixed(2) + ' s';
  $('contacts').textContent = s.contacts;
  $('seedvalue').textContent = s.seed;
  $('play').textContent = s.running ? 'Ⅱ Pause physics' : '▶ Start physics';
  $('step').disabled = !connected || s.running;
  $('connection').textContent = s.running ? '● Physics running' : '● Connected · paused';
  const demo = s.demo;
  const task = s.task;
  $('progress').hidden = !demo && !task;
  $('progress').value = task?.progress || demo?.progress || 0;
  if (task) {
    const metrics = task.name === 'drawer-open'
      ? 'Drawer opened ' + (task.drawer_travel_m * 100).toFixed(1) + ' cm · Grip contact ' + task.bilateral_contact_s.toFixed(1) + ' s'
      : 'Lift ' + (task.peak_lift_m * 100).toFixed(1) + ' cm · Placement error ' + (task.placement_error_m * 1000).toFixed(1) + ' mm';
    $('runstatus').textContent = task.status === 'running'
      ? (s.running ? '' : 'Paused · ') + task.phase + (task.attempt > 1 ? ' · Retry ' + (task.attempt - 1) : '') + ' · ' + Math.round(task.progress * 100) + '% · ' + metrics
      : task.status.toUpperCase() + ' · ' + (task.reason || metrics + ' · Released and stable.') + ' · ' + task.controller;
  } else if (demo?.status === 'running') {
    $('runstatus').textContent = (s.running ? '' : 'Paused · ') + demo.phase + ' · ' + Math.round(demo.progress * 100) + '%';
  } else if (demo?.status === 'completed') {
    $('runstatus').textContent = 'Arm demo completed · Left tip moved ' + (demo.peak_tip_displacement_m.left * 100).toFixed(1) + ' cm · Right tip moved ' + (demo.peak_tip_displacement_m.right * 100).toFixed(1) + ' cm. Run again or use the joint controls.';
  } else {
    $('runstatus').textContent = s.running ? 'Physics running · Motors hold their targets. Select Pick & place cup to execute the physical skill.' : 'Paused · Pick & place cup runs the physical teacher from a fresh selected seed.';
  }
  if (s.last_motion) {
    const m = s.last_motion;
    $('motion').textContent = (m.accepted ? 'Moving to target. ' : 'Target unreachable. ') + 'IK residual: ' + (m.ik_residual_m * 1000).toFixed(1) + ' mm · Actual tip error: ' + (m.actual_error_m * 1000).toFixed(1) + ' mm';
  } else $('motion').textContent = '';
  if (!$('joints').children.length) controls();
  for (const j of s.joints) {
    const output = $('value-' + j.name), input = $(j.name);
    if (output) output.textContent = j.actual.toFixed(2) + ' → ' + j.target.toFixed(2);
    if (input && document.activeElement !== input) input.value = j.target;
  }
}

async function frame() {
  const response = await request('frame?camera=' + $('camera').value);
  const next = URL.createObjectURL(await response.blob());
  const image = new Image();
  image.src = next;
  try { await image.decode(); } catch (error) { URL.revokeObjectURL(next); error.offline = true; throw error; }
  $('feed').src = next;
  if (frameURL) URL.revokeObjectURL(frameURL);
  frameURL = next;
  lastFrame = performance.now();
  freshFrame = false;
}

function controls() {
  const parent = $('joints'); parent.replaceChildren();
  for (const joint of state.joints.filter(j => j.name.startsWith(arm + '_'))) {
    const row = document.createElement('div'); row.className = 'joint';
    const label = document.createElement('label'); label.htmlFor = joint.name;
    label.textContent = joint.name.slice(arm.length + 1).replaceAll('_', ' ');
    const output = document.createElement('output'); output.id = 'value-' + joint.name;
    label.append(output);
    const input = document.createElement('input'); input.type = 'range'; input.id = joint.name;
    input.min = joint.range[0]; input.max = joint.range[1]; input.step = 'any'; input.value = joint.target;
    input.disabled = !connected;
    input.onchange = () => {
      const value = Number(input.value);
      action(async () => {
        display(await api('control', {name: joint.name, value})); freshFrame = true;
        message('Motor movement started. Pause stops the simulation.');
      });
    };
    row.append(label, input); parent.append(row);
  }
}

function action(fn) {
  pending++;
  queue = queue.then(async () => {
    try { await fn(); }
    catch (error) { if (error.offline) availability(false); message(error.message, true); }
    finally { pending--; }
  });
  return queue;
}

$('demo').onclick = () => action(async () => {
  display(await api('demo', {seed: Number($('seed').value)})); freshFrame = true;
  message('Running a real motor-motion demonstration. This does not yet grasp or place objects.');
});
$('pickup').onclick = () => action(async () => {
  display(await api('task', {name: 'cup-place', seed: Number($('seed').value)})); freshFrame = true;
  message('Executing the physical cup skill. Contact, lift and released placement are checked; this is not camera-based AI yet.');
});
$('drawer').onclick = () => action(async () => {
  display(await api('task', {name: 'drawer-open', seed: Number($('seed').value)})); freshFrame = true;
  message('Left-arm drawer task in its own cabinet workcell. Opening through verified handle contact is required.');
});
$('fork').onclick = () => action(async () => {
  display(await api('task', {name: 'fork-retrieve', seed: Number($('seed').value)})); freshFrame = true;
  message('Left arm opens the drawer, then retrieves the fork in the SAME episode with no reset. 10/10 on held-out seeds 40-49.');
});
$('plate').onclick = () => action(async () => {
  display(await api('task', {name: 'plate-place-left', seed: Number($('seed').value)})); freshFrame = true;
  message('Left arm grasps the plate by its raised rim and lifts it. The carry to the marker is still being finished; the reported status is honest.');
});
$('bimanual').onclick = () => action(async () => {
  display(await api('task', {name: 'bimanual-plate-lift', seed: Number($('seed').value)})); freshFrame = true;
  message('Both arms pinch opposing rim segments and carry the plate level to its mat. 10/10 on held-out seeds 50-59, open-gripper control 0/4.');
});
$('execute').onclick = () => action(async () => {
  display(await api('execute', {instruction: $('instruction').value, seed: Number($('seed').value)})); freshFrame = true;
  message('Observing the cup from camera pixels. Unsupported commands are refused without starting motion.');
});
for (const [id, name] of [['learned', 'learned-cup-place'], ['hybrid', 'learned-cup-safe-place']]) {
  $(id).onclick = () => action(async () => {
    message('Loading the local ACT policy…');
    display(await api('task', {name, seed: Number($('seed').value)})); freshFrame = true;
    message('Experimental camera-to-action rollout. Success is checked from physical lift and released placement.');
  });
}
$('play').onclick = () => action(async () => {
  display(await api('playback', {running: !state.running})); freshFrame = true;
});
$('step').onclick = () => action(async () => {
  display(await api('step', {steps: 50})); freshFrame = true;
});
$('reset').onclick = () => action(async () => {
  display(await api('reset', {seed: Number($('seed').value)})); freshFrame = true;
  $('planresult').replaceChildren(); message('Reset and paused at seed ' + state.seed);
});
$('reach').onclick = () => action(async () => {
  const target = [...state.end_effectors[arm]]; target[2] += .03;
  display(await api('reach', {arm, target})); freshFrame = true;
});
$('camera').onchange = () => { freshFrame = true; };
$('retry').onclick = () => action(refresh);
document.querySelectorAll('[data-arm]').forEach(button => button.onclick = () => {
  arm = button.dataset.arm;
  document.querySelectorAll('[data-arm]').forEach(x => x.classList.toggle('active', x === button));
  if (state) { controls(); display(state); }
});
$('plan').onclick = () => action(async () => {
  $('planresult').replaceChildren();
  const result = await api('plan', {instruction: $('instruction').value});
  for (const skill of result.skills) {
    const li = document.createElement('li');
    li.textContent = skill.name.replaceAll('_', ' ') + ' · ' + skill.object + ' · ' + skill.arm + (skill.target ? ' → ' + skill.target : '');
    $('planresult').append(li);
  }
  message('Preview only; this plan has not executed. Camera commands support the cup only; experimental ACT has separate controls.');
});

async function refresh() {
  const next = await api('state');
  if (!connected || freshFrame || next.running || next.time !== state?.time || performance.now() - lastFrame > 1500) await frame();
  availability(true); display(next);
}
async function tick() {
  if (!pending) await action(refresh);
  setTimeout(tick, connected ? 100 : 1500);
}
availability(false); tick();
