/**
 * ORCA: Marine Ecosystem Reasoning with Collaborative Agents
 * Master Frontend Controller & Analytics Engine
 */

// Application State
const ORCA = {
  currentSection: 'dashboard',
  map: null,
  mapLayers: { stations: null, species: null },
  charts: {},
  lastAnalysisResult: null,
  stationsData: [],
  speciesData: [],
  trendsData: null
};

// Preset Scenarios for demonstration & scenario testing
const PRESETS = {
  optimal: {
    name: "Optimal Kelp Forest (Euphotic)",
    depth: 15.0,
    temperature: 14.8,
    salinity: 33.45,
    dissolved_oxygen: 6.20,
    chlorophyll: 1.85,
    phosphate: 0.55,
    nitrate: 5.20,
    silicate: 6.50,
    latitude: 34.25,
    longitude: -119.85,
    month: 5
  },
  depression: {
    name: "Sub-surface Oxygen Depression (Mesopelagic)",
    depth: 85.0,
    temperature: 9.8,
    salinity: 33.95,
    dissolved_oxygen: 3.20,
    chlorophyll: 0.25,
    phosphate: 1.45,
    nitrate: 16.50,
    silicate: 18.20,
    latitude: 33.50,
    longitude: -118.80,
    month: 6
  },
  hypoxic: {
    name: "Severe Hypoxia / Dead Zone Basin",
    depth: 140.0,
    temperature: 8.6,
    salinity: 34.22,
    dissolved_oxygen: 1.15,
    chlorophyll: 0.04,
    phosphate: 2.35,
    nitrate: 29.50,
    silicate: 38.00,
    latitude: 33.70,
    longitude: -119.50,
    month: 7
  },
  heatwave: {
    name: "Marine Heatwave & Phytoplankton Bloom",
    depth: 5.0,
    temperature: 22.4,
    salinity: 33.60,
    dissolved_oxygen: 4.80,
    chlorophyll: 11.80,
    phosphate: 0.40,
    nitrate: 2.10,
    silicate: 4.00,
    latitude: 33.20,
    longitude: -117.40,
    month: 8
  }
};

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPresetButtons();
  initFormHandlers();
  loadDataSummary();
  loadTrends();
  initMap();
  loadRecentReports();
  loadBiodiversityData();

  // Run initial optimal analysis to populate dashboard cards
  loadScenario('optimal');
  runPipelineAnalysis();
});

/* -------------------------------------------------------------
   Navigation & Section Router
------------------------------------------------------------- */
function initNavigation() {
  const links = document.querySelectorAll('.nav-link');
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-section');
      switchSection(target);
    });
  });
}

function switchSection(sectionId) {
  ORCA.currentSection = sectionId;
  
  // Update links
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const activeLink = document.querySelector(`.nav-link[data-section="${sectionId}"]`);
  if (activeLink) activeLink.classList.add('active');

  // Update sections
  document.querySelectorAll('.app-section').forEach(sec => {
    sec.style.display = 'none';
  });
  const activeSec = document.getElementById(`section-${sectionId}`);
  if (activeSec) {
    activeSec.style.display = 'block';
  }

  // Section specific triggers
  if (sectionId === 'map' && ORCA.map) {
    setTimeout(() => ORCA.map.invalidateSize(), 200);
  } else if (sectionId === 'trends') {
    renderTrendsCharts();
  } else if (sectionId === 'biodiversity') {
    renderBiodiversityCharts();
  }
}

/* -------------------------------------------------------------
   Preset Scenario Selection
------------------------------------------------------------- */
function initPresetButtons() {
  const container = document.getElementById('preset-container');
  if (!container) return;

  container.innerHTML = '';
  Object.keys(PRESETS).forEach(key => {
    const p = PRESETS[key];
    const btn = document.createElement('button');
    btn.className = 'scenario-pill';
    btn.innerHTML = `<i class="fa-solid fa-water"></i> ${p.name}`;
    btn.onclick = () => {
      document.querySelectorAll('.scenario-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadScenario(key);
    };
    container.appendChild(btn);
  });
}

function loadScenario(key) {
  const data = PRESETS[key];
  if (!data) return;

  const fields = ['depth', 'temperature', 'salinity', 'dissolved_oxygen', 'chlorophyll', 'nitrate', 'phosphate', 'silicate', 'latitude', 'longitude', 'month'];
  fields.forEach(f => {
    const el = document.getElementById(`input-${f}`);
    if (el) el.value = data[f];
  });
}

function getFormValues() {
  return {
    depth: parseFloat(document.getElementById('input-depth').value) || 10.0,
    temperature: parseFloat(document.getElementById('input-temperature').value) || 14.0,
    salinity: parseFloat(document.getElementById('input-salinity').value) || 33.4,
    dissolved_oxygen: parseFloat(document.getElementById('input-dissolved_oxygen').value) || 5.0,
    chlorophyll: parseFloat(document.getElementById('input-chlorophyll').value) || 1.5,
    phosphate: parseFloat(document.getElementById('input-phosphate').value) || 0.7,
    nitrate: parseFloat(document.getElementById('input-nitrate').value) || 8.0,
    silicate: parseFloat(document.getElementById('input-silicate').value) || 8.0,
    latitude: parseFloat(document.getElementById('input-latitude').value) || 34.0,
    longitude: parseFloat(document.getElementById('input-longitude').value) || -120.0,
    month: parseInt(document.getElementById('input-month').value) || 6
  };
}

/* -------------------------------------------------------------
   Interactive Analysis & Orchestration Pipeline Execution
------------------------------------------------------------- */
function initFormHandlers() {
  const runBtn = document.getElementById('btn-run-pipeline');
  if (runBtn) {
    runBtn.onclick = () => runPipelineAnalysis();
  }

  const quickPredBtn = document.getElementById('btn-quick-predict');
  if (quickPredBtn) {
    quickPredBtn.onclick = () => runQuickPrediction();
  }

  const uploadInput = document.getElementById('csv-file-upload');
  if (uploadInput) {
    uploadInput.onchange = handleFileUpload;
  }
}

async function runPipelineAnalysis() {
  const values = getFormValues();
  const runBtn = document.getElementById('btn-run-pipeline');
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Reasoning in Progress...';
  }

  // Reset pipeline UI nodes to active/running state
  document.querySelectorAll('.pipeline-node').forEach(node => {
    node.className = 'pipeline-node';
    const sub = node.querySelector('.node-sub');
    if (sub) sub.innerText = 'Evaluating...';
  });

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values)
    });

    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    ORCA.lastAnalysisResult = data;

    // Render results
    renderAnalysisResults(data);
    updateDashboardCards(data);
    loadRecentReports();
  } catch (err) {
    console.error('Pipeline error:', err);
    alert('Pipeline analysis error: ' + err.message);
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run ORCA Reasoning Pipeline';
    }
  }
}

function renderAnalysisResults(data) {
  // Update Pipeline Track Nodes
  data.pipeline_steps.forEach(step => {
    const nodeEl = document.getElementById(`node-step-${step.step_number}`);
    if (nodeEl) {
      nodeEl.className = `pipeline-node ${step.status === 'alert' ? 'alert' : 'completed'}`;
      const sub = nodeEl.querySelector('.node-sub');
      if (sub) sub.innerText = `${step.execution_time_ms} ms`;
    }
  });

  // Final Assessment Badge & Summary
  const assessBadge = document.getElementById('analysis-final-assessment');
  if (assessBadge) {
    assessBadge.innerText = data.final_assessment;
    assessBadge.className = 'badge ' + (
      data.final_assessment.includes('Healthy') ? 'badge-healthy' :
      data.final_assessment.includes('Moderate') ? 'badge-warning' : 'badge-critical'
    );
  }

  const confEl = document.getElementById('analysis-confidence-val');
  if (confEl) {
    confEl.innerText = `${(data.final_confidence * 100).toFixed(1)}%`;
  }

  const timeEl = document.getElementById('analysis-exec-time');
  if (timeEl) {
    timeEl.innerText = `${data.total_execution_time_ms} ms`;
  }

  // Render Detailed Agent Findings Container
  const findingsContainer = document.getElementById('agent-findings-list');
  if (findingsContainer) {
    findingsContainer.innerHTML = '';
    const agentMap = data.agent_results || {};

    Object.keys(agentMap).forEach(k => {
      const ag = agentMap[k];
      const card = document.createElement('div');
      card.className = 'orca-card';
      card.style.marginBottom = '14px';
      card.style.padding = '16px';

      let findingsHtml = '';
      (ag.findings || []).forEach(f => {
        findingsHtml += `
          <div class="agent-finding-item ${f.status}">
            <div class="finding-header">
              <span class="finding-title">${f.indicator.replace(/_/g, ' ').toUpperCase()}</span>
              <span class="badge badge-${f.status}">${f.observed_value}</span>
            </div>
            <div class="finding-desc">${f.description}</div>
            ${f.baseline_reference ? `<div style="font-size:11px; color:#64748b; margin-top:3px;">Ref: ${f.baseline_reference}</div>` : ''}
          </div>
        `;
      });

      card.innerHTML = `
        <div class="orca-card-header">
          <div class="orca-card-title">
            <i class="fa-solid fa-microchip" style="color:var(--accent-cyan)"></i> ${ag.display_name}
          </div>
          <span class="badge badge-${ag.severity}">${ag.severity.toUpperCase()}</span>
        </div>
        <p style="font-size:12.5px; color:#cbd5e1; margin-bottom:12px;">${ag.summary}</p>
        ${findingsHtml}
      `;
      findingsContainer.appendChild(card);
    });
  }

  // Update Reasoning Steps
  const stepsContainer = document.getElementById('reasoning-steps-list');
  if (stepsContainer) {
    stepsContainer.innerHTML = '';
    const coordMetrics = data.agent_results?.coordinator_reasoning?.metrics || {};
    (coordMetrics.reasoning_steps || []).forEach(step => {
      const li = document.createElement('li');
      li.style.marginBottom = '8px';
      li.style.fontSize = '12.5px';
      li.style.lineHeight = '1.5';
      li.innerText = step;
      stepsContainer.appendChild(li);
    });
  }

  // Update Report View
  const reportViewer = document.getElementById('markdown-report-viewer');
  if (reportViewer) {
    reportViewer.innerHTML = renderMarkdownToHtml(data.report_markdown);
  }
}

function updateDashboardCards(data) {
  const coord = data.agent_results?.coordinator_reasoning?.metrics || {};
  const risk = data.agent_results?.risk_analysis?.metrics || {};
  const ml = data.agent_results?.ml_analysis?.metrics || {};
  const bio = data.agent_results?.marine_biodiversity?.metrics || {};

  // Ecosystem Status
  const statusEl = document.getElementById('dash-ecosystem-status');
  if (statusEl) {
    statusEl.innerText = data.final_assessment.split('(')[0].trim();
  }

  // ML Prediction
  const mlPredEl = document.getElementById('dash-ml-prediction');
  if (mlPredEl) {
    mlPredEl.innerText = ml.prediction || 'Healthy';
  }

  // Risk Level
  const riskEl = document.getElementById('dash-risk-level');
  if (riskEl) {
    riskEl.innerText = risk.overall_risk_level || 'LOW';
  }

  // Species Richness
  const richEl = document.getElementById('dash-species-richness');
  if (richEl) {
    richEl.innerText = bio.species_richness || '22';
  }

  // Update Probabilities Chart on ML section
  renderProbabilitiesChart(ml.probabilities || { "Healthy": 0.95, "Moderate Stress": 0.04, "High Stress": 0.01 });
}

/* -------------------------------------------------------------
   Interactive Mapping (Leaflet.js)
------------------------------------------------------------- */
function initMap() {
  const mapEl = document.getElementById('marine-map');
  if (!mapEl) return;

  // Center on California Current (Monterey Bay to Santa Barbara)
  ORCA.map = L.map('marine-map').setView([34.5, -120.5], 7);

  // High-contrast Dark Matter Basemap
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 18
  }).addTo(ORCA.map);

  loadStationsOnMap();
}

async function loadStationsOnMap() {
  try {
    const res = await fetch('/api/stations?limit=400');
    if (!res.ok) return;
    const stations = await res.json();
    ORCA.stationsData = stations;

    const layerGroup = L.layerGroup();

    stations.forEach(st => {
      const color = st.ecosystem_condition === 'Healthy' ? '#10b981' :
                    st.ecosystem_condition === 'Moderate Stress' ? '#f59e0b' : '#ef4444';

      const circle = L.circleMarker([st.latitude, st.longitude], {
        radius: 6,
        fillColor: color,
        color: '#070c18',
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.8
      });

      const popupHtml = `
        <div style="font-size:12px; line-height:1.4;">
          <strong style="color:#38bdf8;">CalCOFI Station #${st.station_id}</strong><br>
          <b>Condition:</b> <span style="color:${color}; font-weight:bold;">${st.ecosystem_condition}</span><br>
          <b>Depth:</b> ${st.depth} m | <b>Temp:</b> ${st.temperature}°C<br>
          <b>Salinity:</b> ${st.salinity} PSU | <b>DO:</b> ${st.dissolved_oxygen} ml/L<br>
          <b>Chlorophyll:</b> ${st.chlorophyll} μg/L<br>
          <button class="orca-btn btn-primary" style="margin-top:8px; padding:4px 8px; font-size:11px;" 
                  onclick="loadStationToAnalysis(${st.temperature}, ${st.salinity}, ${st.dissolved_oxygen}, ${st.chlorophyll}, ${st.depth}, ${st.latitude}, ${st.longitude})">
            Load into ORCA Analysis
          </button>
        </div>
      `;
      circle.bindPopup(popupHtml);
      layerGroup.addLayer(circle);
    });

    layerGroup.addTo(ORCA.map);
    ORCA.mapLayers.stations = layerGroup;
  } catch (err) {
    console.error('Error loading stations:', err);
  }
}

window.loadStationToAnalysis = function(temp, sal, doVal, chla, depth, lat, lon) {
  document.getElementById('input-temperature').value = temp;
  document.getElementById('input-salinity').value = sal;
  document.getElementById('input-dissolved_oxygen').value = doVal;
  document.getElementById('input-chlorophyll').value = chla;
  document.getElementById('input-depth').value = depth;
  document.getElementById('input-latitude').value = lat;
  document.getElementById('input-longitude').value = lon;
  switchSection('analysis');
  runPipelineAnalysis();
};

/* -------------------------------------------------------------
   Charts & Visualizations (Chart.js)
------------------------------------------------------------- */
async function loadTrends() {
  try {
    const res = await fetch('/api/trends');
    if (!res.ok) return;
    ORCA.trendsData = await res.json();
  } catch (e) {
    console.error('Trends fetch error:', e);
  }
}

function renderTrendsCharts() {
  if (!ORCA.trendsData || !document.getElementById('chart-temperature-trend')) return;

  const data = ORCA.trendsData;

  // Temperature Chart
  renderLineChart('chart-temperature-trend', data.years, data.temperature, 'Mean Annual Temp (°C)', '#f43f5e');
  
  // Dissolved Oxygen Chart
  renderLineChart('chart-oxygen-trend', data.years, data.dissolved_oxygen, 'Mean Dissolved Oxygen (ml/L)', '#06b6d4');
  
  // Chlorophyll-a Chart
  renderLineChart('chart-chlorophyll-trend', data.years, data.chlorophyll, 'Mean Chlorophyll-a (μg/L)', '#10b981');
  
  // Ecological Stress Index
  renderLineChart('chart-stress-trend', data.years, data.stress_index, 'Ecological Stress Index (0-1)', '#f59e0b');
}

function renderLineChart(canvasId, labels, values, label, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  if (ORCA.charts[canvasId]) {
    ORCA.charts[canvasId].destroy();
  }

  ORCA.charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: label,
        data: values,
        borderColor: color,
        backgroundColor: color + '20',
        borderWidth: 2.5,
        pointRadius: 3,
        tension: 0.35,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

function renderProbabilitiesChart(probMap) {
  const ctx = document.getElementById('chart-ml-probabilities');
  if (!ctx) return;

  if (ORCA.charts['chart-ml-probabilities']) {
    ORCA.charts['chart-ml-probabilities'].destroy();
  }

  const labels = Object.keys(probMap);
  const values = labels.map(l => (probMap[l] * 100).toFixed(1));
  const colors = ['#10b981', '#f59e0b', '#ef4444'];

  ORCA.charts['chart-ml-probabilities'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw}% Probability` } }
      },
      scales: {
        y: { max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#f1f5f9' } }
      }
    }
  });
}

/* -------------------------------------------------------------
   Biodiversity Explorer
------------------------------------------------------------- */
async function loadBiodiversityData() {
  try {
    const statsRes = await fetch('/api/biodiversity-stats');
    if (statsRes.ok) {
      const stats = await statsRes.json();
      ORCA.biodiversityStats = stats;
    }

    const spRes = await fetch('/api/species?limit=100');
    if (spRes.ok) {
      ORCA.speciesData = await spRes.json();
      renderSpeciesTable(ORCA.speciesData);
    }
  } catch (e) {
    console.error('Biodiversity fetch error:', e);
  }
}

function renderBiodiversityCharts() {
  if (!ORCA.biodiversityStats) return;

  const phyla = ORCA.biodiversityStats.phyla || {};
  const ctx = document.getElementById('chart-phyla-breakdown');
  if (ctx) {
    if (ORCA.charts['chart-phyla-breakdown']) ORCA.charts['chart-phyla-breakdown'].destroy();

    ORCA.charts['chart-phyla-breakdown'] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: Object.keys(phyla),
        datasets: [{
          data: Object.values(phyla),
          backgroundColor: ['#06b6d4', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#64748b'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { color: '#cbd5e1', font: { size: 11 } } }
        }
      }
    });
  }
}

function renderSpeciesTable(speciesList) {
  const tbody = document.getElementById('species-table-body');
  if (!tbody) return;

  tbody.innerHTML = '';
  speciesList.slice(0, 30).forEach(sp => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${sp.scientificName}</strong><br><span style="color:#64748b; font-size:11px;">${sp.vernacularName || 'Marine Organism'}</span></td>
      <td><span class="badge badge-info">${sp.phylum}</span></td>
      <td>${sp.class || 'N/A'}</td>
      <td>${sp.depth ? sp.depth + ' m' : 'Euphotic'}</td>
      <td>${sp.decimalLatitude.toFixed(2)}°N, ${sp.decimalLongitude.toFixed(2)}°W</td>
    `;
    tbody.appendChild(tr);
  });
}

/* -------------------------------------------------------------
   Data Summary & Platform Provenance
------------------------------------------------------------- */
async function loadDataSummary() {
  try {
    const res = await fetch('/api/data-summary');
    if (!res.ok) return;
    const summary = await res.json();

    const envEl = document.getElementById('stat-env-records');
    if (envEl) envEl.innerText = summary.total_environmental_records.toLocaleString();

    const bioEl = document.getElementById('stat-bio-records');
    if (bioEl) bioEl.innerText = summary.total_biodiversity_occurrences.toLocaleString();

    const spEl = document.getElementById('stat-species-count');
    if (spEl) spEl.innerText = summary.distinct_marine_species.toLocaleString();

    const phylaEl = document.getElementById('stat-phyla-count');
    if (phylaEl) phylaEl.innerText = summary.distinct_phyla.toLocaleString();
  } catch (e) {
    console.error('Data summary error:', e);
  }
}

/* -------------------------------------------------------------
   Reports & Sessions
------------------------------------------------------------- */
async function loadRecentReports() {
  try {
    const res = await fetch('/api/reports?limit=10');
    if (!res.ok) return;
    const reports = await res.json();

    const tbody = document.getElementById('recent-reports-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';
    reports.forEach(r => {
      const tr = document.createElement('tr');
      const badgeClass = r.final_assessment.includes('Healthy') ? 'badge-healthy' :
                         r.final_assessment.includes('Moderate') ? 'badge-warning' : 'badge-critical';

      tr.innerHTML = `
        <td><strong style="color:var(--accent-sky);">${r.session_id}</strong></td>
        <td>${r.timestamp.split('T')[0]}</td>
        <td>${r.coordinates} (${r.depth})</td>
        <td><span class="badge ${badgeClass}">${r.final_assessment}</span></td>
        <td>${r.confidence}</td>
        <td>
          <button class="orca-btn btn-secondary" style="padding:4px 8px; font-size:11px;" onclick="viewReport('${r.session_id}')">
            <i class="fa-solid fa-eye"></i> View
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error('Recent reports error:', e);
  }
}

window.viewReport = async function(sessionId) {
  try {
    const res = await fetch(`/api/report/${sessionId}`);
    if (!res.ok) return;
    const reportData = await res.json();
    
    switchSection('reports');
    const viewer = document.getElementById('markdown-report-viewer');
    if (viewer) {
      viewer.innerHTML = renderMarkdownToHtml(reportData.report_markdown);
    }
  } catch (e) {
    console.error('View report error:', e);
  }
};

window.downloadReportMarkdown = function() {
  if (!ORCA.lastAnalysisResult?.report_markdown) {
    alert('Please run an analysis first to generate a report.');
    return;
  }
  const blob = new Blob([ORCA.lastAnalysisResult.report_markdown], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ORCA_Assessment_Report_${Date.now()}.md`;
  a.click();
};

window.printReport = function() {
  window.print();
};

/* -------------------------------------------------------------
   CSV Upload Handler
------------------------------------------------------------- */
async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const statusEl = document.getElementById('upload-status-msg');
  if (statusEl) statusEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Parsing oceanographic CSV...';

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Failed to upload file');
    const data = await res.json();

    if (statusEl) {
      statusEl.innerHTML = `
        <div style="color:var(--accent-emerald); font-size:12px; margin-top:8px;">
          ✓ Uploaded ${data.filename} (${data.total_records} records found). 
          Evaluated sample of ${data.sample_evaluations.length} records successfully.
        </div>
      `;
    }
  } catch (err) {
    if (statusEl) statusEl.innerHTML = `<span style="color:var(--accent-rose);">Upload error: ${err.message}</span>`;
  }
}

/* -------------------------------------------------------------
   Markdown to HTML Parser Helper
------------------------------------------------------------- */
function renderMarkdownToHtml(md) {
  if (!md) return '<p>No report generated yet. Run an analysis above.</p>';

  let html = md
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>')
    .replace(/\*\*(.*?)\*\*/gim, '<b>$1</b>')
    .replace(/\*(.*?)\*/gim, '<i>$1</i>')
    .replace(/\n\n/gim, '<br><br>')
    .replace(/^- (.*$)/gim, '<li>$1</li>');

  // Convert markdown tables
  html = html.replace(/\|(.+)\|/gim, (match) => {
    const cells = match.split('|').filter(c => c.trim() !== '');
    if (cells.some(c => c.includes('---'))) return '';
    return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
  });

  return html;
}
