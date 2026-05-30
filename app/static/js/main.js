// Shared utilities for the Log Clustering Service UI

// Relative time formatter
function relativeTime(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// Poll health endpoint and show model status
async function checkModelStatus() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (!data.model_loaded) {
      const banner = document.createElement('div');
      banner.className = 'model-warning';
      banner.innerHTML = `
        <strong>⚠ Model not loaded.</strong>
        Run <code>python ml/train.py</code> inside the container or POST <code>/api/v1/retrain</code>.
      `;
      banner.style.cssText = `
        position:fixed; bottom:1rem; right:1rem; background:#3a2a10; border:1px solid #f59e0b44;
        color:#f59e0b; padding:.75rem 1rem; border-radius:8px; font-size:.82rem; z-index:999;
        max-width:380px;
      `;
      document.body.appendChild(banner);
    }
  } catch {}
}

checkModelStatus();
