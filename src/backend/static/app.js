/* ═══════════════════════════════════════════════════════════════════════════
   DeepRetail Enterprise Controller & Video Analysis Suite
   Zero Emojis • Vector Icons • Video Upload & Source Switching
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
  'use strict';

  // ─── STATE ───────────────────────────────────────────────────────────────
  const state = {
    sessionId: 'demo_session',
    audioEnabled: true,
    audioCtx: null,
    cartItems: [],
    cartTotalQty: 0,        // track qty for beep-on-change
    subtotal: 0.0,
    tax: 0.0,
    grandTotal: 0.0,
    activeSource: 0,
    activeMode: 'camera',
    ws: null,
    isCheckoutOpen: false,
  };

  // Enterprise product catalog metadata (SKUs, labels, unit prices)
  const PRODUCT_CATALOG = {
    'chings_manchurian': { sku: 'SKU-CHM-01', label: "Ching's Manchurian Soup", price: 10.0 },
    'chings_hakka': { sku: 'SKU-CHK-02', label: "Ching's Hakka Noodles", price: 10.0 },
    'homelite_matchbox': { sku: 'SKU-HLM-03', label: 'Homelite Safety Matches', price: 2.0 },
    'vaseline_jelly': { sku: 'SKU-VSJ-04', label: 'Vaseline Petroleum Jelly', price: 50.0 },
  };

  // ─── DOM SELECTORS ────────────────────────────────────────────────────────
  const DOM = {
    liveFeedImg: document.getElementById('liveFeedImg'),
    viewportLoader: document.getElementById('viewportLoader'),
    aiStatusText: document.getElementById('aiStatusText'),
    activeSourceLabel: document.getElementById('activeSourceLabel'),
    telemetryFps: document.getElementById('telemetryFps'),
    telemetryFrame: document.getElementById('telemetryFrame'),
    telemetryTracks: document.getElementById('telemetryTracks'),
    soundToggleBtn: document.getElementById('soundToggleBtn'),
    soundIconOn: document.getElementById('soundIconOn'),
    soundIconOff: document.getElementById('soundIconOff'),
    resetSystemBtn: document.getElementById('resetSystemBtn'),
    tabKioskBtn: document.getElementById('tabKioskBtn'),
    tabOperationsBtn: document.getElementById('tabOperationsBtn'),
    modeCameraBtn: document.getElementById('modeCameraBtn'),
    modeVideoBtn: document.getElementById('modeVideoBtn'),
    videoSuitePanel: document.getElementById('videoSuitePanel'),
    uploadDropzone: document.getElementById('uploadDropzone'),
    videoFileInput: document.getElementById('videoFileInput'),
    videoChipsList: document.getElementById('videoChipsList'),
    currentSessionId: document.getElementById('currentSessionId'),
    newCustomerBtn: document.getElementById('newCustomerBtn'),
    cartEmptyState: document.getElementById('cartEmptyState'),
    cartList: document.getElementById('cartList'),
    billSubtotal: document.getElementById('billSubtotal'),
    billTax: document.getElementById('billTax'),
    billGrandTotal: document.getElementById('billGrandTotal'),
    billItemCount: document.getElementById('billItemCount'),
    checkoutBtn: document.getElementById('checkoutBtn'),
    targetsList: document.getElementById('targetsList'),
    operationsModal: document.getElementById('operationsModal'),
    operationsCloseBtn: document.getElementById('operationsCloseBtn'),
    operationsBackdrop: document.getElementById('operationsBackdrop'),
    opInventoryTableBody: document.getElementById('opInventoryTableBody'),
    opInventoryCount: document.getElementById('opInventoryCount'),
    auditLogList: document.getElementById('auditLogList'),
    receiptModal: document.getElementById('receiptModal'),
    modalScrim: document.getElementById('modalScrim'),
    receiptModalClose: document.getElementById('receiptModalClose'),
    posReceipt: document.getElementById('posReceipt'),
    posInvoiceId: document.getElementById('posInvoiceId'),
    posTimestamp: document.getElementById('posTimestamp'),
    posItems: document.getElementById('posItems'),
    posSubtotal: document.getElementById('posSubtotal'),
    posTax: document.getElementById('posTax'),
    posGrandTotal: document.getElementById('posGrandTotal'),
    upiAmountTag: document.getElementById('upiAmountTag'),
    printReceiptBtn: document.getElementById('printReceiptBtn'),
    finishShopperBtn: document.getElementById('finishShopperBtn'),
    btnStreamVideoLive: document.getElementById('btnStreamVideoLive'),
    btnBatchProcessVideo: document.getElementById('btnBatchProcessVideo'),
    batchProgressBox: document.getElementById('batchProgressBox'),
    batchProgressTitle: document.getElementById('batchProgressTitle'),
    batchProgressPct: document.getElementById('batchProgressPct'),
    batchProgressBarFill: document.getElementById('batchProgressBarFill'),
    batchProgressMeta: document.getElementById('batchProgressMeta'),
    offlineBillModal: document.getElementById('offlineBillModal'),
    offlineModalScrim: document.getElementById('offlineModalScrim'),
    offlineBillModalClose: document.getElementById('offlineBillModalClose'),
    offlineBadgeFilename: document.getElementById('offlineBadgeFilename'),
    offFramesVal: document.getElementById('offFramesVal'),
    offDurationVal: document.getElementById('offDurationVal'),
    offPicksVal: document.getElementById('offPicksVal'),
    offReturnsVal: document.getElementById('offReturnsVal'),
    offNetItemsVal: document.getElementById('offNetItemsVal'),
    offInvoiceId: document.getElementById('offInvoiceId'),
    offDateStr: document.getElementById('offDateStr'),
    offItemsList: document.getElementById('offItemsList'),
    offReturnsContainer: document.getElementById('offReturnsContainer'),
    offReturnsList: document.getElementById('offReturnsList'),
    offSubtotal: document.getElementById('offSubtotal'),
    offTax: document.getElementById('offTax'),
    offGrandTotal: document.getElementById('offGrandTotal'),
    offBarcodeNum: document.getElementById('offBarcodeNum'),
    offTimelineList: document.getElementById('offTimelineList'),
    btnPrintOfflineReceipt: document.getElementById('btnPrintOfflineReceipt'),
    btnSyncOfflineToCart: document.getElementById('btnSyncOfflineToCart'),
    toastStack: document.getElementById('toastStack'),
  };

  // ─── WEB AUDIO API SYNTHESIZER ─────────────────────────────────────────────
  function initAudio() {
    if (!state.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) state.audioCtx = new AudioCtx();
    }
    if (state.audioCtx && state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
  }

  function playPickChime() {
    if (!state.audioEnabled || !state.audioCtx) return;
    try {
      const ctx = state.audioCtx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.18, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.23);
    } catch (e) { /* ignore */ }
  }

  function playReturnChime() {
    if (!state.audioEnabled || !state.audioCtx) return;
    try {
      const ctx = state.audioCtx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(310, ctx.currentTime + 0.16);
      gain.gain.setValueAtTime(0.14, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.21);
    } catch (e) { /* ignore */ }
  }

  function playCheckoutFanfare() {
    if (!state.audioEnabled || !state.audioCtx) return;
    try {
      const ctx = state.audioCtx;
      const notes = [523.25, 659.25, 783.99, 1046.50];
      notes.forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, ctx.currentTime + idx * 0.08);
        gain.gain.setValueAtTime(0.16, ctx.currentTime + idx * 0.08);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + idx * 0.08 + 0.25);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(ctx.currentTime + idx * 0.08);
        osc.stop(ctx.currentTime + idx * 0.08 + 0.26);
      });
    } catch (e) { /* ignore */ }
  }

  // ─── HIGH-SPEED VIDEO STREAM ───────────────────────────────────────────────
  let isFetchingFrame = false;
  function updateLiveFrame() {
    if (isFetchingFrame) return;
    isFetchingFrame = true;

    const img = new Image();
    img.src = `/frame/latest?t=${Date.now()}`;

    img.onload = () => {
      DOM.liveFeedImg.src = img.src;
      DOM.viewportLoader.style.display = 'none';
      isFetchingFrame = false;
    };

    img.onerror = () => {
      isFetchingFrame = false;
    };
  }

  // ─── WEBSOCKET CONNECTION ──────────────────────────────────────────────────
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      DOM.aiStatusText.textContent = 'EDGE AI ACTIVE';
    };

    state.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleLiveEvent(data);
      } catch (err) {
        console.error('[WS] Parse error:', err);
      }
    };

    state.ws.onclose = () => {
      DOM.aiStatusText.textContent = 'CONNECTING...';
      setTimeout(connectWebSocket, 2000);
    };

    state.ws.onerror = () => {
      try { state.ws.close(); } catch (e) {}
    };
  }

  function handleLiveEvent(data) {
    if (data.event === 'pick') {
      playPickChime();
      fetchCart();
      appendAuditLog('PICK', data.product, data.timestamp || Date.now() / 1000);
      showToast('pick', `Added ${cleanName(data.product)} to cart`);
    } else if (data.event === 'return') {
      playReturnChime();
      fetchCart();
      appendAuditLog('RETURN', data.product, data.timestamp || Date.now() / 1000);
      showToast('return', `Removed ${cleanName(data.product)} from bag`);
    } else if (data.event === 'anomaly' || data.type === 'alert') {
      const alertData = data.data || data;
      showToast('alert', alertData.message || 'Security Anomaly detected');
      appendAuditLog('ANOMALY', alertData.message || 'Alert', Date.now() / 1000);
    }
  }

  function cleanName(raw) {
    const cat = PRODUCT_CATALOG[raw];
    if (cat) return cat.label;
    return raw.replace(/_/g, ' ').toUpperCase();
  }

  // ─── TELEMETRY & STATUS ────────────────────────────────────────────────────
  async function fetchTelemetry() {
    try {
      const res = await fetch('/pipeline/status');
      if (!res.ok) return;
      const data = await res.json();
      DOM.telemetryFps.textContent = (data.fps || 0).toFixed(1);
      DOM.telemetryFrame.textContent = `#${data.frame_id || 0}`;
      DOM.telemetryTracks.textContent = data.active_tracks || 0;

      if (data.session_id && data.session_id !== 'none' && data.session_id !== state.sessionId) {
        state.sessionId = data.session_id;
        DOM.currentSessionId.textContent = state.sessionId;
        fetchCart();
      }

      DOM.aiStatusText.textContent = data.running ? 'EDGE AI ACTIVE' : 'PIPELINE IDLE';
    } catch (e) { /* ignore */ }
  }

  // ─── CART & BILLING ────────────────────────────────────────────────────────
  async function fetchCart() {
    try {
      const res = await fetch(`/cart/${state.sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      const items = data.items || [];
      const newQty = items.reduce((acc, it) => acc + (it.quantity || 1), 0);
      const oldQty = state.cartTotalQty;
      renderCart(items, data.total || 0.0);
      // Beep whenever the cart quantity changes (catches events missed by WS)
      if (newQty > oldQty) {
        initAudio();
        playPickChime();
      } else if (newQty < oldQty && oldQty > 0) {
        initAudio();
        playReturnChime();
      }
      state.cartTotalQty = newQty;
    } catch (e) { /* ignore */ }
  }

  function renderCart(items, total) {
    state.cartItems = items;
    state.subtotal = total;
    state.tax = +(total * 0.05).toFixed(2);
    state.grandTotal = +(state.subtotal + state.tax).toFixed(2);

    DOM.billSubtotal.textContent = `₹${state.subtotal.toFixed(2)}`;
    DOM.billTax.textContent = `₹${state.tax.toFixed(2)}`;
    DOM.billGrandTotal.textContent = `₹${state.grandTotal.toFixed(2)}`;

    const totalQty = items.reduce((acc, it) => acc + (it.quantity || 1), 0);
    DOM.billItemCount.textContent = `${totalQty} item${totalQty === 1 ? '' : 's'}`;

    if (items.length === 0) {
      DOM.cartEmptyState.style.display = 'flex';
      DOM.cartList.style.display = 'none';
      DOM.checkoutBtn.disabled = true;
      updateTargetBadges([]);
    } else {
      DOM.cartEmptyState.style.display = 'none';
      DOM.cartList.style.display = 'flex';
      DOM.checkoutBtn.disabled = false;

      DOM.cartList.innerHTML = items.map(it => {
        const cat = PRODUCT_CATALOG[it.name] || {
          sku: 'SKU-GENERIC',
          label: it.name.replace(/_/g, ' ').toUpperCase(),
        };
        return `
          <div class="cart-row">
            <div class="row-product-info">
              <h4>${cat.label}</h4>
              <div class="row-product-meta">${cat.sku} • ₹${it.unit_price.toFixed(2)} unit</div>
            </div>
            <div class="row-pricing">
              <span class="badge-qty">Qty ${it.quantity}</span>
              <span class="row-subtotal mono">₹${it.subtotal.toFixed(2)}</span>
            </div>
          </div>
        `;
      }).join('');

      updateTargetBadges(items.map(it => it.name));
    }
  }

  function updateTargetBadges(names) {
    if (!names || names.length === 0) {
      DOM.targetsList.innerHTML = '<span class="empty-hint">Standby • Ready for product placement</span>';
      return;
    }
    DOM.targetsList.innerHTML = names.map(n => {
      const cat = PRODUCT_CATALOG[n] || { label: n };
      return `
        <span class="target-badge">
          <span class="dot"></span>
          <span>${cat.label}</span>
          <span style="color:var(--accent-emerald);">BILLED</span>
        </span>
      `;
    }).join('');
  }

  // ─── CHECKOUT & THERMAL RECEIPT ────────────────────────────────────────────
  DOM.checkoutBtn.addEventListener('click', () => {
    initAudio();
    openReceiptModal();
  });

  async function openReceiptModal() {
    if (state.cartItems.length === 0) return;
    playCheckoutFanfare();

    const invNum = `INV-${Math.floor(10000 + Math.random() * 90000)}`;
    const now = new Date().toLocaleString();

    DOM.posInvoiceId.textContent = invNum;
    DOM.posTimestamp.textContent = now;
    DOM.posSubtotal.textContent = `₹${state.subtotal.toFixed(2)}`;
    DOM.posTax.textContent = `₹${state.tax.toFixed(2)}`;
    DOM.posGrandTotal.textContent = `₹${state.grandTotal.toFixed(2)}`;
    DOM.upiAmountTag.textContent = `₹${state.grandTotal.toFixed(2)}`;

    DOM.posItems.innerHTML = state.cartItems.map(it => {
      const cat = PRODUCT_CATALOG[it.name] || { label: it.name };
      return `
        <div class="pos-item-row">
          <span>${it.quantity}x ${cat.label.slice(0, 18)}</span>
          <span class="mono">₹${it.subtotal.toFixed(2)}</span>
        </div>
      `;
    }).join('');

    DOM.receiptModal.style.display = 'flex';
    state.isCheckoutOpen = true;

    try {
      await fetch(`/cart/checkout/${state.sessionId}`, { method: 'POST' });
    } catch (e) { /* ignore */ }
  }

  DOM.receiptModalClose.addEventListener('click', () => {
    DOM.receiptModal.style.display = 'none';
    state.isCheckoutOpen = false;
  });
  DOM.modalScrim.addEventListener('click', () => {
    DOM.receiptModal.style.display = 'none';
    state.isCheckoutOpen = false;
  });

  DOM.printReceiptBtn.addEventListener('click', () => {
    window.print();
  });

  DOM.finishShopperBtn.addEventListener('click', () => {
    DOM.receiptModal.style.display = 'none';
    state.isCheckoutOpen = false;
    startNewSession();
  });

  function startNewSession() {
    state.sessionId = `cust_${Math.floor(1000 + Math.random() * 9000)}`;
    DOM.currentSessionId.textContent = state.sessionId;
    renderCart([], 0.0);
    showToast('pick', `New Customer Session: ${state.sessionId}`);
  }

  DOM.newCustomerBtn.addEventListener('click', startNewSession);

  // Receipt Modal Payment Tabs
  document.querySelectorAll('.pay-option').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.pay-option').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      const t = e.target.dataset.tab;
      document.getElementById('paneThermal').style.display = t === 'receipt' ? 'block' : 'none';
      document.getElementById('paneUpi').style.display = t === 'upi' ? 'block' : 'none';
      document.getElementById('paneNfc').style.display = t === 'nfc' ? 'block' : 'none';
    });
  });

  // ─── SOURCE SWITCHING & VIDEO UPLOAD SUITE ──────────────────────────────────
  DOM.modeCameraBtn.addEventListener('click', async () => {
    initAudio();
    DOM.modeCameraBtn.classList.add('active');
    DOM.modeVideoBtn.classList.remove('active');
    DOM.videoSuitePanel.style.display = 'none';
    state.activeMode = 'camera';

    try {
      await fetch('/pipeline/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 0 }),
      });
      DOM.activeSourceLabel.textContent = 'CAMERA 0';
      showToast('pick', 'Video source switched to Live Webcam 0');
    } catch (e) { /* ignore */ }
  });

  DOM.modeVideoBtn.addEventListener('click', () => {
    initAudio();
    DOM.modeVideoBtn.classList.add('active');
    DOM.modeCameraBtn.classList.remove('active');
    DOM.videoSuitePanel.style.display = 'flex';
    state.activeMode = 'video';
    loadUploadedVideos();
  });

  // Upload dropzone interactions
  DOM.uploadDropzone.addEventListener('click', () => {
    DOM.videoFileInput.click();
  });

  DOM.uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    DOM.uploadDropzone.style.borderColor = 'var(--accent-emerald)';
  });

  DOM.uploadDropzone.addEventListener('dragleave', () => {
    DOM.uploadDropzone.style.borderColor = '';
  });

  DOM.uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    DOM.uploadDropzone.style.borderColor = '';
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleVideoUpload(e.dataTransfer.files[0]);
    }
  });

  DOM.videoFileInput.addEventListener('change', () => {
    if (DOM.videoFileInput.files && DOM.videoFileInput.files.length > 0) {
      handleVideoUpload(DOM.videoFileInput.files[0]);
    }
  });

  async function handleVideoUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    showToast('pick', `Uploading video: ${file.name}...`);
    try {
      const res = await fetch('/video/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        showToast('pick', `Video uploaded! Analyzing: ${data.filename}`);
        DOM.activeSourceLabel.textContent = `VIDEO (${data.filename.slice(0, 12)}...)`;
        loadUploadedVideos();
      } else {
        showToast('alert', 'Video upload failed');
      }
    } catch (err) {
      showToast('alert', 'Error uploading video');
    }
  }

  async function loadUploadedVideos() {
    try {
      const res = await fetch('/videos');
      if (!res.ok) return;
      const data = await res.json();
      const list = data.videos || [];

      if (list.length === 0) {
        DOM.videoChipsList.innerHTML = '<span class="no-videos-text">No uploaded videos yet. Upload a video above to test recorded footage.</span>';
        return;
      }

      let selectedChip = null;
      DOM.videoChipsList.innerHTML = list.map(v => `
        <div class="video-chip ${v.is_active ? 'active' : ''}" data-path="${v.path}" data-name="${v.filename}">
          <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          <span>${v.filename}</span>
          <span style="color:var(--text-tertiary);">(${v.size_mb} MB)</span>
        </div>
      `).join('');

      // Auto-select active or first video if none selected
      if (!state.selectedVideoPath && list.length > 0) {
        const activeVid = list.find(v => v.is_active) || list[0];
        state.selectedVideoPath = activeVid.path;
        state.selectedVideoName = activeVid.filename;
      }

      document.querySelectorAll('.video-chip').forEach(chip => {
        if (chip.dataset.path === state.selectedVideoPath) {
          chip.classList.add('selected');
        }
        chip.addEventListener('click', (e) => {
          initAudio();
          const target = e.currentTarget;
          document.querySelectorAll('.video-chip').forEach(c => c.classList.remove('selected'));
          target.classList.add('selected');
          state.selectedVideoPath = target.dataset.path;
          state.selectedVideoName = target.dataset.name;
          showToast('pick', `Selected: ${state.selectedVideoName}`);
        });
      });
    } catch (e) { /* ignore */ }
  }

  // Action Button 1: Stream to Live Viewport
  DOM.btnStreamVideoLive.addEventListener('click', async () => {
    initAudio();
    if (!state.selectedVideoPath) {
      showToast('alert', 'Please upload or select a video recording first');
      return;
    }
    try {
      await fetch('/pipeline/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: state.selectedVideoPath }),
      });
      DOM.activeSourceLabel.textContent = `VIDEO (${state.selectedVideoName.slice(0, 12)}...)`;
      showToast('pick', `Streaming to Live Viewport: ${state.selectedVideoName}`);
      loadUploadedVideos();
    } catch (err) {
      showToast('alert', 'Failed to switch video stream');
    }
  });

  // Action Button 2: Process Entire Video & Generate Final Bill (Offline)
  DOM.btnBatchProcessVideo.addEventListener('click', async () => {
    initAudio();
    if (!state.selectedVideoPath) {
      showToast('alert', 'Please upload or select a shopping recording first');
      return;
    }

    DOM.batchProgressBox.style.display = 'flex';
    DOM.batchProgressBarFill.style.width = '2%';
    DOM.batchProgressPct.textContent = '2%';
    DOM.batchProgressTitle.textContent = `Analyzing ${state.selectedVideoName}...`;
    DOM.batchProgressMeta.textContent = 'Initializing DeepRetail Vision AI...';
    DOM.btnBatchProcessVideo.disabled = true;

    try {
      const res = await fetch('/video/analyze-offline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: state.selectedVideoPath }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || 'Failed to start analysis');
      }
      const data = await res.json();
      pollOfflineJob(data.job_id);
    } catch (err) {
      DOM.btnBatchProcessVideo.disabled = false;
      DOM.batchProgressBox.style.display = 'none';
      showToast('alert', `Analysis failed: ${err.message}`);
    }
  });

  function pollOfflineJob(jobId) {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/video/job-status/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        if (job.status === 'processing' || job.status === 'queued') {
          const pct = Math.max(5, Math.round(job.progress || 0));
          DOM.batchProgressBarFill.style.width = `${pct}%`;
          DOM.batchProgressPct.textContent = `${pct}%`;
          DOM.batchProgressMeta.textContent = `Processed Frame ${job.current_frame || 0} of ${job.total_frames || 0}`;
        } else if (job.status === 'completed') {
          clearInterval(timer);
          DOM.btnBatchProcessVideo.disabled = false;
          DOM.batchProgressBarFill.style.width = '100%';
          DOM.batchProgressPct.textContent = '100%';
          DOM.batchProgressMeta.textContent = 'Analysis complete! Generating final bill...';
          setTimeout(() => {
            DOM.batchProgressBox.style.display = 'none';
            renderOfflineBill(job.result);
          }, 600);
        } else if (job.status === 'failed') {
          clearInterval(timer);
          DOM.btnBatchProcessVideo.disabled = false;
          DOM.batchProgressBox.style.display = 'none';
          showToast('alert', `Batch processing error: ${job.error || 'Unknown error'}`);
        }
      } catch (e) {
        clearInterval(timer);
        DOM.btnBatchProcessVideo.disabled = false;
        DOM.batchProgressBox.style.display = 'none';
      }
    }, 500);
  }

  function renderOfflineBill(result) {
    if (!result || !result.bill) return;
    const b = result.bill;

    DOM.offlineBadgeFilename.textContent = result.video_filename || 'Recorded Video';
    DOM.offFramesVal.textContent = result.total_frames || 0;
    const durSec = result.duration_sec || 0;
    const mm = String(Math.floor(durSec / 60)).padStart(2, '0');
    const ss = String(Math.floor(durSec % 60)).padStart(2, '0');
    DOM.offDurationVal.textContent = `${mm}:${ss}`;
    DOM.offPicksVal.textContent = result.total_picks || 0;
    DOM.offReturnsVal.textContent = result.total_returns || 0;
    DOM.offNetItemsVal.textContent = result.net_items_count || 0;

    DOM.offInvoiceId.textContent = b.invoice_id;
    DOM.offDateStr.textContent = b.date;

    // Items Billed
    if (!b.items || b.items.length === 0) {
      DOM.offItemsList.innerHTML = '<div style="padding:10px 0; color:#888; text-align:center;">No merchandise purchased in this session.</div>';
    } else {
      DOM.offItemsList.innerHTML = b.items.map(it => `
        <div class="rit-row">
          <span>${it.name}</span>
          <span style="text-align:center;">${it.quantity}</span>
          <span style="text-align:right;">₹${it.unit_price.toFixed(2)}</span>
          <span class="rit-tot">₹${it.line_total.toFixed(2)}</span>
        </div>
      `).join('');
    }

    // Returned Items (Deducted from bill)
    if (b.returns && b.returns.length > 0) {
      DOM.offReturnsContainer.style.display = 'block';
      DOM.offReturnsList.innerHTML = b.returns.map(it => `
        <div class="returned-row">
          <span>${it.name}</span>
          <span style="text-align:center;">-${it.quantity}</span>
          <span style="text-align:right;">₹${it.unit_price.toFixed(2)}</span>
          <span style="text-align:right; font-weight:700;">-₹${it.deducted_total.toFixed(2)}</span>
        </div>
      `).join('');
    } else {
      DOM.offReturnsContainer.style.display = 'none';
    }

    DOM.offSubtotal.textContent = `₹${b.subtotal.toFixed(2)}`;
    DOM.offTax.textContent = `₹${b.tax.toFixed(2)}`;
    DOM.offGrandTotal.textContent = `₹${b.grand_total.toFixed(2)}`;
    DOM.offBarcodeNum.textContent = `890${b.invoice_id.replace(/\D/g, '').padEnd(10, '7')}`;

    // Timeline Log
    const tl = result.timeline || [];
    if (tl.length === 0) {
      DOM.offTimelineList.innerHTML = '<div style="font-size:11px; color:var(--text-tertiary);">No product movements detected in video.</div>';
    } else {
      DOM.offTimelineList.innerHTML = tl.map(ev => `
        <div class="timeline-item">
          <span class="tl-time mono">${ev.time}</span>
          <span class="tl-badge ${ev.event}">${ev.event}</span>
          <span style="flex:1; font-weight:600;">${ev.name}</span>
          <span class="mono" style="color:var(--text-secondary);">₹${ev.price.toFixed(2)}</span>
        </div>
      `).join('');
    }

    DOM.offlineBillModal.style.display = 'flex';
    playSuccessChime();
    showToast('pick', `Generated Final Bill: ${b.invoice_id} • Total: ₹${b.grand_total.toFixed(2)}`);
  }

  // Offline Bill Modal Controls
  DOM.offlineBillModalClose.addEventListener('click', () => {
    DOM.offlineBillModal.style.display = 'none';
  });
  DOM.offlineModalScrim.addEventListener('click', () => {
    DOM.offlineBillModal.style.display = 'none';
  });
  DOM.btnPrintOfflineReceipt.addEventListener('click', () => {
    window.print();
  });
  DOM.btnSyncOfflineToCart.addEventListener('click', () => {
    DOM.offlineBillModal.style.display = 'none';
  });

  // ─── ZONE CALIBRATOR ───────────────────────────────────────────────────────
  const ZONE_PRESETS = {
    'Right Side (Standard)': {
      shelf_roi: [20, 20, 310, 460],
      basket_roi: [330, 20, 620, 460],
      shelf_slots: {},
    },
    'Left Side': {
      shelf_roi: [330, 20, 620, 460],
      basket_roi: [20, 20, 310, 460],
      shelf_slots: {},
    },
    'Bottom Half': {
      shelf_roi: [20, 20, 620, 240],
      basket_roi: [20, 250, 620, 460],
      shelf_slots: {},
    },
  };

  document.querySelectorAll('.zone-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      initAudio();
      document.querySelectorAll('.zone-btn').forEach(b => b.classList.remove('active'));
      const presetKey = e.target.dataset.preset;
      e.target.classList.add('active');

      const presetData = ZONE_PRESETS[presetKey];
      if (presetData) {
        try {
          await fetch('/zones/preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(presetData),
          });
          showToast('pick', `Bag Zone Set: ${presetKey}`);
        } catch (err) { /* ignore */ }
      }
    });
  });

  // ─── AUDIO TOGGLE ──────────────────────────────────────────────────────────
  DOM.soundToggleBtn.addEventListener('click', () => {
    initAudio();
    state.audioEnabled = !state.audioEnabled;
    DOM.soundIconOn.style.display = state.audioEnabled ? 'block' : 'none';
    DOM.soundIconOff.style.display = state.audioEnabled ? 'none' : 'block';
    showToast('pick', state.audioEnabled ? 'Audio Chimes Enabled' : 'Audio Chimes Muted');
  });

  // ─── RESET SYSTEM ──────────────────────────────────────────────────────────
  DOM.resetSystemBtn.addEventListener('click', async () => {
    initAudio();
    if (!confirm('Reset stock levels from CSV and clear carts?')) return;
    try {
      const res = await fetch('/admin/reset', { method: 'POST' });
      if (res.ok) {
        showToast('pick', 'Inventory and carts restored');
        startNewSession();
        fetchInventory();
      }
    } catch (e) { /* ignore */ }
  });

  // ─── OPERATIONS SLIDE OVER DRAWER ──────────────────────────────────────────
  DOM.tabOperationsBtn.addEventListener('click', () => {
    DOM.operationsModal.style.display = 'flex';
    fetchInventory();
  });

  DOM.operationsCloseBtn.addEventListener('click', closeOperations);
  DOM.operationsBackdrop.addEventListener('click', closeOperations);

  function closeOperations() {
    DOM.operationsModal.style.display = 'none';
  }

  async function fetchInventory() {
    try {
      const res = await fetch('/inventory');
      if (!res.ok) return;
      const data = await res.json();
      const prods = data.products || [];
      const lowNames = new Set((data.low_stock || []).map(p => p.name));

      DOM.opInventoryCount.textContent = `${prods.length} Products`;
      DOM.opInventoryTableBody.innerHTML = prods.map(p => {
        const isLow = lowNames.has(p.name);
        const cat = PRODUCT_CATALOG[p.name] || { label: p.name, sku: 'SKU' };
        return `
          <tr>
            <td><strong>${cat.label}</strong></td>
            <td style="color:var(--text-tertiary);">${p.category}</td>
            <td><strong style="color:${isLow ? 'var(--accent-rose)' : 'inherit'};">${p.stock} units</strong></td>
            <td class="mono">₹${p.price.toFixed(2)}</td>
            <td>
              <span class="tag-status ${isLow ? 'low' : 'ok'}">
                ${isLow ? 'Low Stock Warning' : 'Optimal Stock'}
              </span>
            </td>
          </tr>
        `;
      }).join('');
    } catch (e) { /* ignore */ }
  }

  function appendAuditLog(action, details, timestamp) {
    const timeStr = new Date(timestamp * 1000).toLocaleTimeString();
    const item = document.createElement('div');
    item.className = `audit-entry ${action.toLowerCase()}`;
    item.innerHTML = `
      <div><strong>${action}</strong>: <span>${details}</span></div>
      <span class="mono" style="color:var(--text-tertiary); font-size:11px;">${timeStr}</span>
    `;

    DOM.auditLogList.prepend(item);
    if (DOM.auditLogList.children.length > 25) {
      DOM.auditLogList.removeChild(DOM.auditLogList.lastChild);
    }
  }

  // ─── MINIMALIST TOASTS ─────────────────────────────────────────────────────
  function showToast(type, msg) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <div style="font-weight:700; color:var(--text-primary);">${msg}</div>
    `;
    DOM.toastStack.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      toast.style.transition = 'all 0.2s ease';
      setTimeout(() => toast.remove(), 200);
    }, 3200);
  }

  // ─── INITIALIZATION ────────────────────────────────────────────────────────
  // Eagerly init audio so context is ready before first pick arrives
  // (browsers allow AudioContext creation; it just starts suspended until a gesture)
  initAudio();
  window.addEventListener('click', initAudio, { once: true });
  window.addEventListener('keydown', initAudio, { once: true });

  connectWebSocket();
  fetchCart();
  fetchTelemetry();

  setInterval(updateLiveFrame, 120);
  setInterval(fetchTelemetry, 600);
  setInterval(fetchCart, 800);
})();
