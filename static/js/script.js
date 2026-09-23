/**
 * AgriSense - Modern Frontend Controller
 * Handles dynamic crop search, live unit conversions, interactive Chart.js visualizations,
 * and responsive form interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
  initCropSearch();
  initYearFutureIndicator();
  initComparisonChart();
  initDashboardCharts();
  initQuickPills();
});

/**
 * 1. Searchable Dynamic Crop Selector & Mode Controller
 */
let cropCategoriesData = null;

function initCropSearch() {
  const searchInput = document.getElementById('cropSearchInput');
  const dropdown = document.getElementById('cropResultsDropdown');
  const hiddenCropInput = document.getElementById('selectedCropInput');
  const isManualModeInput = document.getElementById('isManualModeInput');
  const selectedCropDisplay = document.getElementById('selectedCropDisplay');
  const selectedCropTypeBadge = document.getElementById('selectedCropTypeBadge');

  // Mode switcher tabs
  const tabCatalog = document.getElementById('tabCatalogMode');
  const tabManual = document.getElementById('tabManualMode');
  const catalogSection = document.getElementById('catalogModeSection');
  const manualSection = document.getElementById('manualModeSection');
  const manualInput = document.getElementById('manualCropInput');
  const manualCategorySelect = document.getElementById('manualCategorySelect');
  const applyManualBtn = document.getElementById('applyManualCropBtn');

  // Load category taxonomy for live filtering
  fetch('/api/crops')
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        cropCategoriesData = data.categories || [];
      }
    })
    .catch(err => console.log('Crop categories load notice:', err));

  // Mode tab switching
  if (tabCatalog && tabManual) {
    tabCatalog.addEventListener('click', () => {
      tabCatalog.classList.add('active');
      tabManual.classList.remove('active');
      catalogSection.style.display = 'block';
      manualSection.style.display = 'none';
      if (isManualModeInput) isManualModeInput.value = '0';
      if (searchInput && searchInput.value) {
        selectCrop(searchInput.value.trim(), false);
      }
    });

    tabManual.addEventListener('click', () => {
      tabManual.classList.add('active');
      tabCatalog.classList.remove('active');
      catalogSection.style.display = 'none';
      manualSection.style.display = 'block';
      if (isManualModeInput) isManualModeInput.value = '1';
      if (manualInput) {
        if (!manualInput.value && hiddenCropInput) {
          manualInput.value = hiddenCropInput.value;
        }
        manualInput.focus();
        if (manualInput.value.trim()) {
          selectCrop(manualInput.value.trim(), true);
        }
      }
    });
  }

  // Manual input dynamic listener
  if (manualInput) {
    manualInput.addEventListener('input', () => {
      const val = manualInput.value.trim();
      if (val) {
        selectCrop(val, true);
      }
    });
  }

  if (applyManualBtn && manualInput) {
    applyManualBtn.addEventListener('click', () => {
      const val = manualInput.value.trim();
      if (val) {
        selectCrop(val, true);
        applyManualBtn.innerHTML = '<i class="fa-solid fa-check-double me-1"></i> Crop Confirmed!';
        setTimeout(() => {
          applyManualBtn.innerHTML = '<i class="fa-solid fa-circle-check me-1"></i> Use Custom Crop';
        }, 1500);
      } else {
        manualInput.focus();
      }
    });
  }

  // Category filter pills
  const catPills = document.querySelectorAll('.cat-pill');
  catPills.forEach(pill => {
    pill.addEventListener('click', () => {
      catPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const selectedCat = pill.getAttribute('data-category');
      filterQuickPillsByCategory(selectedCat);
    });
  });

  if (!searchInput || !dropdown) return;

  let debounceTimer;

  searchInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();

    debounceTimer = setTimeout(() => {
      fetchCropMatches(query);
    }, 150);
  });

  searchInput.addEventListener('focus', () => {
    fetchCropMatches(searchInput.value.trim());
  });

  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.style.display = 'none';
    }
  });

  function fetchCropMatches(query) {
    fetch(`/api/crops/search?q=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => {
        if (!data.success) return;
        renderDropdown(data.data.results, query);
      })
      .catch(err => console.error('Error searching crops:', err));
  }

  function renderDropdown(crops, query) {
    dropdown.innerHTML = '';

    // Render matches
    if (crops && crops.length > 0) {
      crops.slice(0, 10).forEach(crop => {
        const item = document.createElement('div');
        item.className = 'crop-result-item';
        item.innerHTML = `
          <span><i class="fa-solid fa-seedling text-success me-2"></i><strong>${highlightMatch(crop, query)}</strong></span>
          <span class="badge bg-light text-dark border">Select Crop</span>
        `;
        item.addEventListener('click', () => {
          selectCrop(crop, false);
        });
        dropdown.appendChild(item);
      });
    }

    // Always provide "Use as custom manual crop" option if query is entered
    if (query && query.length >= 2) {
      const customItem = document.createElement('div');
      customItem.className = 'crop-result-item manual-action';
      customItem.innerHTML = `
        <span><i class="fa-solid fa-plus-circle me-2"></i>Use "<strong>${query}</strong>" as manual custom crop</span>
        <span class="badge bg-success text-white">Manual Custom</span>
      `;
      customItem.addEventListener('click', () => {
        selectCrop(query, true);
        if (manualInput) manualInput.value = query;
      });
      dropdown.appendChild(customItem);
    } else if (!crops || crops.length === 0) {
      dropdown.innerHTML = `
        <div class="crop-result-item" style="color: #64748b; cursor: default;">
          <span>Type to search 40+ crops or enter manually...</span>
        </div>
      `;
    }

    dropdown.style.display = 'block';
  }

  function selectCrop(crop, isManual = false) {
    if (hiddenCropInput) hiddenCropInput.value = crop;
    if (isManualModeInput) isManualModeInput.value = isManual ? '1' : '0';
    if (!isManual && searchInput) searchInput.value = crop;
    if (isManual && manualInput) manualInput.value = crop;

    if (selectedCropDisplay) {
      selectedCropDisplay.innerText = crop;
      selectedCropDisplay.parentElement.style.display = 'inline-flex';
    }

    if (selectedCropTypeBadge) {
      if (isManual) {
        selectedCropTypeBadge.className = 'badge bg-warning bg-opacity-20 text-dark border border-warning px-2 py-1 rounded small';
        selectedCropTypeBadge.innerHTML = '<i class="fa-solid fa-pen-to-square me-1 text-warning"></i> Custom Manual Crop';
      } else {
        selectedCropTypeBadge.className = 'badge bg-light text-dark border px-2 py-1 rounded small';
        selectedCropTypeBadge.innerHTML = '<i class="fa-solid fa-circle-check me-1 text-success"></i> Library Crop';
      }
    }

    dropdown.style.display = 'none';

    // Update active state on quick pills
    document.querySelectorAll('.crop-pill').forEach(pill => {
      if (pill.getAttribute('data-crop') === crop) {
        pill.classList.add('active');
      } else {
        pill.classList.remove('active');
      }
    });
  }

  function highlightMatch(text, query) {
    if (!query) return text;
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<span style="color:#059669; text-decoration: underline;">$1</span>');
  }

  function filterQuickPillsByCategory(category) {
    const pills = document.querySelectorAll('#cropQuickPills .crop-pill');
    if (!pills.length) return;

    if (category === 'All' || !cropCategoriesData) {
      pills.forEach((p, idx) => {
        p.style.display = (idx < 14) ? 'inline-block' : 'none';
      });
      return;
    }

    const catObj = cropCategoriesData.find(c => c.name === category);
    const validCrops = catObj ? catObj.crops : [];

    pills.forEach(pill => {
      const c = pill.getAttribute('data-crop');
      if (validCrops.includes(c)) {
        pill.style.display = 'inline-block';
      } else {
        pill.style.display = 'none';
      }
    });
  }

  // Expose selectCrop for pills
  window.agriSelectCrop = selectCrop;
}

/**
 * 2. Quick Select Pills for Popular Crops
 */
function initQuickPills() {
  const pills = document.querySelectorAll('.crop-pill');

  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      const crop = pill.getAttribute('data-crop');
      if (window.agriSelectCrop) {
        window.agriSelectCrop(crop, false);
      }
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
    });
  });
}

/**
 * 3. Year & Future-Year Dynamic Indicator
 */
function initYearFutureIndicator() {
  const yearInput = document.getElementById('yearInput');
  const futureBadge = document.getElementById('futureNoticeBadge');

  if (!yearInput || !futureBadge) return;

  function updateYearNotice() {
    const val = parseInt(yearInput.value, 10);
    if (val > 2013) {
      futureBadge.innerHTML = `
        <span class="badge-pill badge-future">
          <i class="fa-solid fa-clock-rotate-left"></i> Future-Year Model Estimate (${val})
        </span>
        <small class="d-block text-muted mt-1" style="font-size: 0.78rem;">
          Year is outside the 1990–2013 historical training period. Yield is estimated using trained statistical patterns.
        </small>
      `;
      futureBadge.style.display = 'block';
    } else if (val < 1990) {
      futureBadge.innerHTML = `
        <span class="badge-pill badge-accent">
          <i class="fa-solid fa-history"></i> Pre-1990 Historical Extrapolation (${val})
        </span>
      `;
      futureBadge.style.display = 'block';
    } else {
      futureBadge.innerHTML = `
        <span class="badge-pill badge-trained">
          <i class="fa-solid fa-circle-check"></i> Within Training Range (1990–2013)
        </span>
      `;
      futureBadge.style.display = 'block';
    }
  }

  yearInput.addEventListener('input', updateYearNotice);
  updateYearNotice();
}

/**
 * 4. Interactive Multi-Crop Comparison Chart
 */
function initComparisonChart() {
  const canvas = document.getElementById('comparisonBarChart');
  if (!canvas) return;

  const dataEl = document.getElementById('comparisonDataJson');
  if (!dataEl) return;

  try {
    const rawData = JSON.parse(dataEl.textContent);
    if (!rawData || rawData.length === 0) return;

    const labels = rawData.map(d => d.crop);
    const predictedKg = rawData.map(d => d.yield_kg_ha);
    const historicalKg = rawData.map(d => d.historical_avg_kg_ha);

    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Predicted Yield (kg/ha)',
            data: predictedKg,
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderColor: '#059669',
            borderWidth: 1.5,
            borderRadius: 6
          },
          {
            label: 'Historical Benchmark (kg/ha)',
            data: historicalKg,
            backgroundColor: 'rgba(245, 158, 11, 0.75)',
            borderColor: '#d97706',
            borderWidth: 1.5,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const val = context.raw;
                return `≈ ${(val / 1000).toFixed(2)} tonnes/ha`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Yield (kg/ha)' }
          }
        }
      }
    });
  } catch (e) {
    console.error('Error rendering comparison chart:', e);
  }
}

/**
 * 5. Analytics Dashboard Charts (8 Real Data Charts)
 */
function initDashboardCharts() {
  const chartContainer = document.getElementById('dashboardChartsContainer');
  if (!chartContainer) return;

  fetch('/api/chart-data')
    .then(res => res.json())
    .then(response => {
      if (!response.success || !response.data) return;
      const data = response.data;

      // Chart 1: Average Yield by Crop
      renderChart('chart1_crop_yield', {
        type: 'bar',
        data: {
          labels: data.chart1_crop_yield.labels,
          datasets: [{
            label: 'Average Yield (kg/ha)',
            data: data.chart1_crop_yield.yield_kg_ha,
            backgroundColor: 'rgba(16, 185, 129, 0.75)',
            borderColor: '#059669',
            borderWidth: 1.5,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      });

      // Chart 2: Historical Yield Trend over Time
      renderChart('chart2_yearly_trend', {
        type: 'line',
        data: {
          labels: data.chart2_yearly_trend.labels,
          datasets: [{
            label: 'Mean Global Yield (kg/ha)',
            data: data.chart2_yearly_trend.yield_kg_ha,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.15)',
            fill: true,
            tension: 0.35,
            pointRadius: 4,
            pointBackgroundColor: '#064e3b'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { beginAtZero: false } }
        }
      });

      // Chart 3: Feature Importance
      renderChart('chart3_feature_importance', {
        type: 'doughnut',
        data: {
          labels: data.chart3_feature_importance.labels,
          datasets: [{
            data: data.chart3_feature_importance.values,
            backgroundColor: [
              '#059669', '#10b981', '#34d399',
              '#f59e0b', '#fbbf24', '#3b82f6'
            ]
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right' }
          }
        }
      });

      // Chart 4: Top 10 Countries by Mean Yield
      renderChart('chart4_top_countries', {
        type: 'bar',
        data: {
          labels: data.chart4_top_countries.labels,
          datasets: [{
            label: 'Average Yield (kg/ha)',
            data: data.chart4_top_countries.yield_kg_ha,
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderColor: '#2563eb',
            borderWidth: 1.5,
            borderRadius: 6
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } }
        }
      });

      // Chart 5: Rainfall vs Yield
      renderChart('chart5_rainfall_vs_yield', {
        type: 'bar',
        data: {
          labels: data.chart5_rainfall_vs_yield.labels,
          datasets: [{
            label: 'Average Yield (kg/ha)',
            data: data.chart5_rainfall_vs_yield.yield_kg_ha,
            backgroundColor: 'rgba(6, 182, 212, 0.75)',
            borderColor: '#0891b2',
            borderWidth: 1.5,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } }
        }
      });

      // Chart 6: Temperature vs Yield
      renderChart('chart6_temp_vs_yield', {
        type: 'bar',
        data: {
          labels: data.chart6_temp_vs_yield.labels,
          datasets: [{
            label: 'Average Yield (kg/ha)',
            data: data.chart6_temp_vs_yield.yield_kg_ha,
            backgroundColor: 'rgba(249, 115, 22, 0.75)',
            borderColor: '#ea580c',
            borderWidth: 1.5,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } }
        }
      });

      // Chart 7: Pesticide Usage vs Yield
      renderChart('chart7_pest_vs_yield', {
        type: 'bar',
        data: {
          labels: data.chart7_pest_vs_yield.labels,
          datasets: [{
            label: 'Average Yield (kg/ha)',
            data: data.chart7_pest_vs_yield.yield_kg_ha,
            backgroundColor: 'rgba(168, 85, 247, 0.75)',
            borderColor: '#9333ea',
            borderWidth: 1.5,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } }
        }
      });

      // Chart 8: Crop Yield Spread (Min, Mean, Max)
      renderChart('chart8_yield_spread', {
        type: 'bar',
        data: {
          labels: data.chart8_yield_spread.labels,
          datasets: [
            {
              label: 'Min Yield',
              data: data.chart8_yield_spread.min_kg_ha,
              backgroundColor: 'rgba(148, 163, 184, 0.7)'
            },
            {
              label: 'Mean Yield',
              data: data.chart8_yield_spread.mean_kg_ha,
              backgroundColor: 'rgba(16, 185, 129, 0.85)'
            },
            {
              label: 'Max Yield',
              data: data.chart8_yield_spread.max_kg_ha,
              backgroundColor: 'rgba(245, 158, 11, 0.75)'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } }
        }
      });
    })
    .catch(err => console.error('Error initializing dashboard charts:', err));

  function renderChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    new Chart(canvas, config);
  }
}
