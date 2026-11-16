const HISTORY_KEY = 'still-checkins-v1';
const ACTIONS = {
  box_breathing: ['Box breathing', 'Breathe in for 4, hold for 4, breathe out for 4, then hold for 4. Repeat gently for a minute.', '◌'],
  grounding: ['Grounding', 'Notice five things you see, four you feel, three you hear, two you smell, and one you taste.', '⌂'],
  deep_work: ['A focused work block', 'Choose one small task, set a gentle timer for 20 minutes, and give it your attention.', '↗'],
  journaling: ['A few lines of journaling', 'Write down what feels most present, without needing to solve it right away.', '✎'],
  rest: ['A real pause', 'Step away for a few minutes, get comfortable, and let your energy come back at its own pace.', '◷'],
  movement: ['Gentle movement', 'Take a short walk or stretch your shoulders and legs. Keep it easy and notice how you feel.', '↝'],
  sound_therapy: ['A calming sound', 'Try a quiet, familiar sound or a few minutes of silence as you wind down.', '♫'],
  light_planning: ['Light planning', 'Write down one or two things that matter today. Leave the rest for another moment.', '☰'],
};
const TIMINGS = { now: 'When you’re ready', within_15_min: 'In the next 15 min', later_today: 'Later today', tonight: 'Tonight' };
const $ = (selector) => document.querySelector(selector);
const form = $('#checkin-form');
const stateColor = { calm: '#78936b', focused: '#82976c', neutral: '#9e9a6b', mixed: '#9c89a7', restless: '#c28b62', overwhelmed: '#bb7968' };

function readHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

function saveHistory(item) {
  const items = [item, ...readHistory()].slice(0, 40);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
  return items;
}

function pretty(value) { return String(value || '').replaceAll('_', ' '); }

function formatDate(dateString, includeTime = false) {
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return 'Just now';
  const options = includeTime ? { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' } : { month: 'short', day: 'numeric' };
  return new Intl.DateTimeFormat(undefined, options).format(date);
}

function showError(message) {
  const error = $('#form-error');
  error.textContent = message;
  error.hidden = false;
}

function renderResult(result) {
  $('#result-empty').hidden = true;
  $('#result-content').hidden = false;
  $('#state-name').textContent = pretty(result.predicted_state);
  $('#state-name').style.color = stateColor[result.predicted_state] || '#58745b';
  const confidence = Math.max(0, Math.min(100, Math.round(result.confidence * 100)));
  $('#confidence-label').textContent = `${confidence}% confidence`;
  $('#confidence-bar').style.width = `${confidence}%`;
  $('#confidence-note').textContent = result.uncertain_flag ? 'The signals point in a few directions.' : 'How strongly the signals point to this read.';
  $('#uncertain-note').hidden = !result.uncertain_flag;
  $('#intensity-value').textContent = result.predicted_intensity;
  const dots = $('#intensity-dots');
  dots.replaceChildren(...Array.from({ length: 5 }, (_, index) => {
    const dot = document.createElement('span');
    dot.className = `intensity-dot${index < result.predicted_intensity ? ' filled' : ''}`;
    return dot;
  }));
  $('#support-message').textContent = result.support_message;
  const [title, description, icon] = ACTIONS[result.what_to_do] || ACTIONS.light_planning;
  $('#action-name').textContent = title;
  $('#action-description').textContent = description;
  $('#action-icon').textContent = icon;
  $('#timing-label').textContent = TIMINGS[result.when_to_do] || pretty(result.when_to_do);
  const scoreList = $('#state-score-list');
  scoreList.replaceChildren(...Object.entries(result.state_scores).map(([state, score]) => {
    const row = document.createElement('div');
    row.className = 'score-row';
    const name = document.createElement('span'); name.textContent = pretty(state);
    const track = document.createElement('span'); track.className = 'score-track';
    const fill = document.createElement('span'); fill.style.width = `${score}%`; track.append(fill);
    const value = document.createElement('strong'); value.textContent = `${score}%`;
    row.append(name, track, value);
    return row;
  }));
}

function historyItem(item, compact = false) {
  const row = document.createElement('div');
  row.className = 'history-item';
  const mark = document.createElement('span'); mark.className = 'history-state-mark'; mark.textContent = '✳';
  mark.style.color = stateColor[item.state] || '#708c61';
  const copy = document.createElement('span'); copy.className = 'history-copy';
  const title = document.createElement('strong'); title.textContent = pretty(item.state);
  const date = document.createElement('span'); date.textContent = formatDate(item.date, !compact);
  copy.append(title, date);
  const intensity = document.createElement('span'); intensity.className = 'history-intensity'; intensity.textContent = `${item.intensity}/5`;
  row.append(mark, copy, intensity);
  return row;
}

function renderHistory() {
  const items = readHistory();
  $('#checkin-count').textContent = items.length;
  const recent = $('#recent-checkins');
  recent.replaceChildren();
  if (!items.length) {
    const empty = document.createElement('div'); empty.className = 'history-empty';
    const icon = document.createElement('span'); icon.className = 'history-empty-icon'; icon.textContent = '◌';
    const text = document.createElement('p'); text.textContent = 'Your check-ins will gather here as you go.';
    empty.append(icon, text); recent.append(empty);
  } else items.slice(0, 3).forEach((item) => recent.append(historyItem(item, true)));

  $('#insight-checkin-total').textContent = items.length;
  const counts = items.reduce((total, item) => { total[item.state] = (total[item.state] || 0) + 1; return total; }, {});
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  $('#insight-top-state').textContent = top ? pretty(top[0]) : '—';
  $('#insight-last-time').textContent = items.length ? formatDate(items[0].date) : '—';
  const full = $('#full-history'); full.replaceChildren();
  if (items.length) items.forEach((item) => full.append(historyItem(item)));
  else {
    const empty = document.createElement('div'); empty.className = 'history-empty';
    const icon = document.createElement('span'); icon.className = 'history-empty-icon'; icon.textContent = '◌';
    const text = document.createElement('p'); text.textContent = 'Your check-ins will show up here. Journal text is never saved.';
    empty.append(icon, text); full.append(empty);
  }
  $('#clear-history').hidden = !items.length;
}

function renderDataset(data) {
  $('#training-samples').textContent = Number(data.training_samples).toLocaleString();
  $('#state-total').textContent = data.states.length;
  $('#chart-total').textContent = Number(data.training_samples).toLocaleString();
  const chart = $('#training-state-chart'); chart.replaceChildren();
  const max = Math.max(...Object.values(data.state_counts));
  Object.entries(data.state_counts).forEach(([state, count]) => {
    const row = document.createElement('div'); row.className = 'bar-row';
    const label = document.createElement('span'); label.className = 'bar-label'; label.textContent = pretty(state);
    const track = document.createElement('span'); track.className = 'bar-track';
    const fill = document.createElement('span'); fill.style.width = `${(count / max) * 100}%`; track.append(fill);
    const value = document.createElement('span'); value.className = 'bar-value'; value.textContent = Number(count).toLocaleString();
    row.append(label, track, value); chart.append(row);
  });
}

function setTimeOfDay() {
  const hour = new Date().getHours();
  const period = hour < 6 ? 'early_morning' : hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : hour < 21 ? 'evening' : 'night';
  $('#time-of-day').value = period;
}

document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => {
  const name = button.dataset.view;
  document.querySelectorAll('.view').forEach((view) => { view.hidden = view.dataset.page !== name; view.classList.toggle('active', view.dataset.page === name); });
  document.querySelectorAll('.nav-link').forEach((link) => link.classList.toggle('active', link.dataset.view === name));
  document.querySelectorAll('.mobile-nav-link').forEach((link) => link.classList.toggle('active', link.dataset.view === name));
  const titles = { checkin: 'DAILY CHECK-IN', insights: 'YOUR PATTERNS', method: 'HOW IT WORKS' };
  $('#current-section').textContent = titles[name];
  window.scrollTo({ top: 0, behavior: 'smooth' });
}));

const RANGE_OUTPUTS = { 'sleep-hours': 'sleep-value', 'energy-level': 'energy-value', 'stress-level': 'stress-value', 'duration-min': 'duration-value' };
document.querySelectorAll('.range-input').forEach((input) => input.addEventListener('input', () => {
  const output = $(`#${RANGE_OUTPUTS[input.id]}`);
  if (output) output.textContent = input.value;
}));

$('#journal-text').addEventListener('input', (event) => {
  const words = event.target.value.trim().split(/\s+/).filter(Boolean).length;
  $('#word-count').textContent = `${words} ${words === 1 ? 'word' : 'words'}`;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  $('#form-error').hidden = true;
  const button = $('#analyze-button');
  button.disabled = true;
  $('#button-label').textContent = 'Listening to the signals…';
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ['sleep_hours', 'energy_level', 'stress_level', 'duration_min']) payload[key] = Number(payload[key]);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The check-in could not be analyzed. Please try again.');
    renderResult(result);
    saveHistory({ state: result.predicted_state, intensity: result.predicted_intensity, confidence: result.confidence, date: new Date().toISOString() });
    renderHistory();
    $('#result-title').textContent = 'Your readout';
    if (window.matchMedia('(max-width: 620px)').matches) $('#result-content').scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (error) { showError(error.message || 'Could not reach the local model. Check that the web app is running.'); }
  finally { button.disabled = false; $('#button-label').textContent = 'Find my next step'; }
});

$('#clear-history').addEventListener('click', () => { localStorage.removeItem(HISTORY_KEY); renderHistory(); });

$('#today-date').textContent = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric' }).format(new Date());
setTimeOfDay();
renderHistory();
fetch('/api/overview').then((response) => { if (!response.ok) throw new Error(); return response.json(); }).then(renderDataset).catch(() => {
  $('#training-samples').textContent = '—';
  $('#chart-total').textContent = '—';
});
