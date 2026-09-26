/**
 * FloraScan AI: Plant Leaf Disease Recognition Application Logic
 * =============================================================
 * Implements:
 * 1. Slide Deck Navigation State Machine (Steps 1 to 10)
 * 2. View Mode Switcher (Slide Deck vs Technical Report vs Live Lab)
 * 3. Canvas-based Procedural Leaf Specimen & Saliency Heatmap (Grad-CAM)
 * 4. Real-time Simulated CNN Inference & Top-3 Diagnostic Probabilities
 * 5. Interactive Chart Renderers (Epoch Curves & Crop Distribution)
 * 6. Interactive Confusion Matrix & CNN Architecture Explorer
 */

document.addEventListener('DOMContentLoaded', () => {

  // =========================================================================
  // 1. DATA DEFINITIONS: 38-CLASS SPECIMEN DATABASE & DIAGNOSTIC PROFILES
  // =========================================================================
  const SPECIMEN_DATABASE = {
    tomato_early_blight: {
      id: 'tomato_early_blight',
      crop: 'Tomato (Solanum lycopersicum)',
      disease: 'Tomato — Early Blight',
      pathogen: 'Alternaria solani (Fungus)',
      type: 'Fungal Infection',
      typeBadgeClass: 'alert',
      confidence: 99.2,
      latency: 14.2,
      top3: [
        { name: 'Tomato: Early Blight', prob: 99.2 },
        { name: 'Tomato: Septoria Leaf Spot', prob: 0.6 },
        { name: 'Tomato: Target Spot', prob: 0.2 }
      ],
      organicRx: 'Apply Bacillus subtilis foliar spray (5g/L) or neem oil emulsion every 7 days. Prune lower infected leaves to prevent soil spore splash.',
      chemicalRx: 'Apply Copper Hydroxide (2g/L) or Chlorothalonil-based protectant fungicide at first lesion appearance; alternate with Azoxystrobin.',
      prevention: 'Avoid overhead sprinkler irrigation. Maintain 45cm plant spacing for adequate air circulation. Implement 3-year solanaceous crop rotation.',
      symptoms: 'Dark brown, concentric target-like rings surrounded by a chlorotic yellow halo on older foliage.',
      lesions: [
        { x: 0.35, y: 0.40, r: 24, halo: 38, rings: 3 },
        { x: 0.62, y: 0.32, r: 18, halo: 30, rings: 2 },
        { x: 0.48, y: 0.65, r: 28, halo: 44, rings: 4 },
        { x: 0.28, y: 0.70, r: 16, halo: 26, rings: 2 }
      ],
      leafType: 'lobed',
      baseColor: '#2d6a4f'
    },
    tomato_late_blight: {
      id: 'tomato_late_blight',
      crop: 'Tomato (Solanum lycopersicum)',
      disease: 'Tomato — Late Blight',
      pathogen: 'Phytophthora infestans (Oomycete)',
      type: 'Oomycete Infection',
      typeBadgeClass: 'danger',
      confidence: 98.7,
      latency: 15.1,
      top3: [
        { name: 'Tomato: Late Blight', prob: 98.7 },
        { name: 'Tomato: Early Blight', prob: 1.0 },
        { name: 'Tomato: Leaf Mold', prob: 0.3 }
      ],
      organicRx: 'Copper octanoate bio-fungicide spray; immediate removal and burial of heavily infected vines to arrest field-wide sporulation.',
      chemicalRx: 'Systemic curative fungicides: Metalaxyl-M (Mefenoxam) + Mancozeb tank mix; spray every 5-7 days under wet, humid overcast conditions.',
      prevention: 'Destroy volunteer tomato and potato culled heaps. Utilize resistant hybrids (e.g., Defiant Ph-R, Mountain Merit).',
      symptoms: 'Large, irregularly shaped, water-soaked dark brown to purplish lesions with pale borders; rapid canopy collapse in cool, moist weather.',
      lesions: [
        { x: 0.42, y: 0.35, r: 42, halo: 55, rings: 0, blotch: true },
        { x: 0.68, y: 0.55, r: 35, halo: 48, rings: 0, blotch: true },
        { x: 0.30, y: 0.60, r: 30, halo: 42, rings: 0, blotch: true }
      ],
      leafType: 'lobed',
      baseColor: '#1b4332'
    },
    corn_common_rust: {
      id: 'corn_common_rust',
      crop: 'Corn / Maize (Zea mays)',
      disease: 'Corn (Maize) — Common Rust',
      pathogen: 'Puccinia sorghi (Basidiomycete Fungus)',
      type: 'Fungal Rust',
      typeBadgeClass: 'alert',
      confidence: 99.5,
      latency: 13.8,
      top3: [
        { name: 'Corn: Common Rust', prob: 99.5 },
        { name: 'Corn: Northern Leaf Blight', prob: 0.3 },
        { name: 'Corn: Cercospora Leaf Spot', prob: 0.2 }
      ],
      organicRx: 'Apply wettable sulfur dust (3g/L) during early vegetative stages. Rotate with potassium silicate soil drenches.',
      chemicalRx: 'Triazole fungicides (Propiconazole, Tebuconazole) or Strobilurin (Pyraclostrobin) when rust pustules cover >5% of the upper canopy prior to silking.',
      prevention: 'Plant rust-resistant certified hybrids with Rp1-D genes. Early planting to avoid peak airborne urediniospore dispersal.',
      symptoms: 'Elongated, cinnamon-brown to dark golden pustules scattered on both leaf surfaces that rupture the epidermis.',
      pustules: [
        { x: 0.40, y: 0.25 }, { x: 0.45, y: 0.32 }, { x: 0.38, y: 0.40 },
        { x: 0.48, y: 0.48 }, { x: 0.42, y: 0.58 }, { x: 0.52, y: 0.65 },
        { x: 0.36, y: 0.72 }, { x: 0.45, y: 0.80 }, { x: 0.55, y: 0.38 }
      ],
      leafType: 'lanceolate',
      baseColor: '#386641'
    },
    potato_early_blight: {
      id: 'potato_early_blight',
      crop: 'Potato (Solanum tuberosum)',
      disease: 'Potato — Early Blight',
      pathogen: 'Alternaria solani (Fungus)',
      type: 'Fungal Blight',
      typeBadgeClass: 'alert',
      confidence: 97.9,
      latency: 14.5,
      top3: [
        { name: 'Potato: Early Blight', prob: 97.9 },
        { name: 'Potato: Late Blight', prob: 1.7 },
        { name: 'Potato: healthy', prob: 0.4 }
      ],
      organicRx: 'Foliar application of compost tea or Serenade (Bacillus amyloliquefaciens). Optimize balanced potassium fertilization to reduce leaf stress.',
      chemicalRx: 'Apply Chlorothalonil or Mancozeb protective sprays starting at row closure; switch to Difenoconazole if lesions advance.',
      prevention: 'Avoid nitrogen deficiency, which predisposes senescence. Allow tubers to mature fully before harvest to prevent spore entry.',
      symptoms: 'Small, dark brown necrotic spots expanding into angular lesions delimited by leaf veins with prominent concentric ridges.',
      lesions: [
        { x: 0.38, y: 0.35, r: 22, halo: 32, rings: 3 },
        { x: 0.58, y: 0.45, r: 26, halo: 38, rings: 3 },
        { x: 0.42, y: 0.65, r: 20, halo: 28, rings: 2 }
      ],
      leafType: 'compound',
      baseColor: '#2d6a4f'
    },
    apple_scab: {
      id: 'apple_scab',
      crop: 'Apple (Malus domestica)',
      disease: 'Apple — Apple Scab',
      pathogen: 'Venturia inaequalis (Ascomycete)',
      type: 'Ascomycete Infection',
      typeBadgeClass: 'alert',
      confidence: 98.9,
      latency: 15.0,
      top3: [
        { name: 'Apple: Apple Scab', prob: 98.9 },
        { name: 'Apple: Cedar Apple Rust', prob: 0.8 },
        { name: 'Apple: Black Rot', prob: 0.3 }
      ],
      organicRx: 'Liquid lime sulfur sprays during dormant and green tip bud stages. Shred or compost autumn leaf litter to eliminate overwintering pseudothecia.',
      chemicalRx: 'Preventive sprays: Captan or Dithianon from bud break to petal fall. Systemic fungicides: Difenoconazole or Cyprodinil post-infection.',
      prevention: 'Prune tree canopy for rapid leaf drying. Plant resistant apple cultivars (e.g., Enterprise, GoldRush, Liberty, Prima).',
      symptoms: 'Dull, olive-green to smoky-brown velvety spots on leaves with indistinct borders, later becoming dark, thickened, and corky.',
      lesions: [
        { x: 0.35, y: 0.38, r: 26, halo: 34, rings: 1, velvety: true },
        { x: 0.62, y: 0.48, r: 32, halo: 42, rings: 1, velvety: true },
        { x: 0.40, y: 0.62, r: 22, halo: 28, rings: 1, velvety: true }
      ],
      leafType: 'ovate',
      baseColor: '#2b5138'
    },
    bell_pepper_healthy: {
      id: 'bell_pepper_healthy',
      crop: 'Bell Pepper (Capsicum annuum)',
      disease: 'Bell Pepper — Healthy Leaf',
      pathogen: 'None (Healthy Plant Tissue)',
      type: 'Healthy Specimen',
      typeBadgeClass: 'success',
      confidence: 99.8,
      latency: 12.9,
      top3: [
        { name: 'Bell Pepper: healthy', prob: 99.8 },
        { name: 'Pepper, bell: Bacterial Spot', prob: 0.1 },
        { name: 'Tomato: healthy', prob: 0.1 }
      ],
      organicRx: 'No pathogen detected. Continue scheduled macro/micronutrient fertigation and drip irrigation maintenance.',
      chemicalRx: 'No chemical intervention necessary. Maintain routine scouting to preserve beneficial predatory insects.',
      prevention: 'Maintain clean tools, weed-free perimeter buffers, and routine soil moisture monitoring to prevent opportunistic stress.',
      symptoms: 'Uniform, vibrant green lamina with intact cellular turgor, smooth cuticle, and no necrotic or chlorotic lesions.',
      lesions: [],
      leafType: 'ovate',
      baseColor: '#10b981'
    }
  };

  // Step Titles for Slide Deck Header
  const STEP_TITLES = [
    "Identify Real-World Problem",
    "Dataset Selection (PlantVillage)",
    "Preprocessing & Data Cleaning",
    "Data Splitting & tf.data Pipeline",
    "CNN Architecture Selection",
    "Model Training Dynamics",
    "Model Performance Evaluation",
    "Hyperparameter Tuning & Regularization",
    "Model Serialization & Multi-Export",
    "Live Demonstration & Field UI"
  ];

  // =========================================================================
  // 2. STATE MANAGEMENT & DOM SELECTION
  // =========================================================================
  let currentSlide = 0;
  let currentSampleId = 'tomato_early_blight';
  let isGradCamActive = false;
  let customUploadedImage = null;

  // Header and View Switcher Elements
  const btnModePresentation = document.getElementById('btn-mode-presentation');
  const btnModeReport = document.getElementById('btn-mode-report');
  const btnModeDemo = document.getElementById('btn-mode-demo');
  const presentationView = document.getElementById('presentation-view');
  const reportView = document.getElementById('report-view');
  const demoSection = document.getElementById('demo-section');
  const btnJumpToLab = document.getElementById('btn-jump-to-lab');

  // Deck Elements
  const slideStepNum = document.getElementById('slide-step-number');
  const slideStepTitle = document.getElementById('slide-step-title');
  const btnPrevSlide = document.getElementById('btn-prev-slide');
  const btnNextSlide = document.getElementById('btn-next-slide');
  const pillBtns = document.querySelectorAll('.pill-btn');
  const slideCards = document.querySelectorAll('.slide-card');

  // Interactive Lab Elements
  const leafCanvas = document.getElementById('leaf-canvas');
  const canvasCtx = leafCanvas ? leafCanvas.getContext('2d') : null;
  const canvasTag = document.getElementById('canvas-tag');
  const btnViewRaw = document.getElementById('btn-view-raw');
  const btnViewGradcam = document.getElementById('btn-view-gradcam');
  const sampleBtns = document.querySelectorAll('.sample-btn');
  const dropZone = document.getElementById('drop-zone');
  const leafFileInput = document.getElementById('leaf-file-input');

  // Diagnosis Card Elements
  const diagBanner = document.getElementById('diagnosis-banner');
  const diagTypePill = document.getElementById('diag-type-pill');
  const diagConfPill = document.getElementById('diag-conf-pill');
  const diagName = document.getElementById('diag-name');
  const diagPathogen = document.getElementById('diag-pathogen');
  const probBarsGroup = document.getElementById('prob-bars-group');
  const rxOrganic = document.getElementById('rx-organic');
  const rxChemical = document.getElementById('rx-chemical');
  const rxPrevention = document.getElementById('rx-prevention');
  const badgeInferenceStatus = document.getElementById('badge-inference-status');

  // CNN Stack Explorer Elements
  const layerBlocks = document.querySelectorAll('.layer-block');
  const layerDetailBox = document.getElementById('layer-detail-box');

  // Copy Code Button
  const btnCopyScript = document.getElementById('btn-copy-script');


  // =========================================================================
  // 3. SLIDE DECK CONTROLLER (STEPS 1 - 10)
  // =========================================================================
  function goToSlide(index) {
    if (index < 0) index = 0;
    if (index >= slideCards.length) index = slideCards.length - 1;
    currentSlide = index;

    // Update slides visibility
    slideCards.forEach((card, idx) => {
      card.classList.toggle('active', idx === currentSlide);
    });

    // Update Step Header
    const formattedNum = String(currentSlide + 1).padStart(2, '0');
    if (slideStepNum) slideStepNum.textContent = formattedNum;
    if (slideStepTitle) slideStepTitle.textContent = STEP_TITLES[currentSlide];

    // Update Pill Buttons
    pillBtns.forEach((btn, idx) => {
      btn.classList.toggle('active', idx === currentSlide);
    });

    // Update prev/next button states
    if (btnPrevSlide) btnPrevSlide.disabled = currentSlide === 0;
    if (btnNextSlide) btnNextSlide.disabled = currentSlide === slideCards.length - 1;

    // Trigger canvas chart renders if visiting respective slides
    if (currentSlide === 1) renderCropDistributionChart();
    if (currentSlide === 5) renderTrainingCurvesChart();
    if (currentSlide === 6) renderConfusionMatrix();
  }

  // Bind deck navigation events
  if (btnPrevSlide) {
    btnPrevSlide.addEventListener('click', () => goToSlide(currentSlide - 1));
  }
  if (btnNextSlide) {
    btnNextSlide.addEventListener('click', () => goToSlide(currentSlide + 1));
  }

  pillBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const slideIndex = parseInt(e.currentTarget.getAttribute('data-slide'), 10);
      goToSlide(slideIndex);
    });
  });

  // Keyboard navigation for slides
  window.addEventListener('keydown', (e) => {
    // Only capture keyboard arrows when active view is presentation
    if (presentationView && presentationView.classList.contains('active')) {
      if (e.key === 'ArrowRight' || e.key === 'PageDown') {
        goToSlide(currentSlide + 1);
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        goToSlide(currentSlide - 1);
      }
    }
  });


  // =========================================================================
  // 4. VIEW MODE SWITCHING (SLIDES vs REPORT vs LAB)
  // =========================================================================
  function setActiveView(mode) {
    if (mode === 'presentation') {
      btnModePresentation.classList.add('active');
      btnModeReport.classList.remove('active');
      btnModeDemo.classList.remove('active');

      presentationView.classList.add('active');
      presentationView.removeAttribute('hidden');
      reportView.classList.remove('active');
      reportView.setAttribute('hidden', '');
    } else if (mode === 'report') {
      btnModePresentation.classList.remove('active');
      btnModeReport.classList.add('active');
      btnModeDemo.classList.remove('active');

      presentationView.classList.remove('active');
      presentationView.setAttribute('hidden', '');
      reportView.classList.add('active');
      reportView.removeAttribute('hidden');
    } else if (mode === 'demo') {
      demoSection.scrollIntoView({ behavior: 'smooth' });
    }
  }

  if (btnModePresentation) {
    btnModePresentation.addEventListener('click', () => setActiveView('presentation'));
  }
  if (btnModeReport) {
    btnModeReport.addEventListener('click', () => setActiveView('report'));
  }
  if (btnModeDemo) {
    btnModeDemo.addEventListener('click', () => setActiveView('demo'));
  }
  if (btnJumpToLab) {
    btnJumpToLab.addEventListener('click', () => {
      demoSection.scrollIntoView({ behavior: 'smooth' });
    });
  }


  // =========================================================================
  // 5. INTERACTIVE CNN ARCHITECTURE EXPLORER (SLIDE 5)
  // =========================================================================
  const LAYER_DESCRIPTIONS = {
    input: "<strong>Input Layer (224 × 224 × 3):</strong> Ingests standard 3-channel RGB leaf photos. Bilinear resizing ensures uniform spatial dimensions.",
    conv1: "<strong>Conv Block 1 (32 Filters, 3×3, ReLU):</strong> Low-level feature extractor. Detects leaf perimeter edges, venation contours, and surface gradients. Batch normalization prevents covariate shift.",
    conv2: "<strong>Conv Block 2 (64 Filters, 3×3, ReLU):</strong> Mid-level texture extractor. Identifies circular chlorosis halos, fungal pustule spots, and color variances.",
    conv3: "<strong>Conv Block 3 (128 Filters, 3×3, ReLU):</strong> High-level feature extractor. Sensitive to concentric target rings (*Alternaria*) and water-soaked lesions (*Phytophthora*).",
    conv4: "<strong>Conv Block 4 (256 Filters, 3×3, ReLU):</strong> Semantic pathology abstraction. Encodes complex disease syndrome fingerprints across varying leaf orientations.",
    dense: "<strong>GlobalAveragePooling2D + Dense(256) + Softmax(38):</strong> Replaces 15M flattened weights with 256 pooled channels. Outputs probability distribution across all 38 target classes."
  };

  layerBlocks.forEach(block => {
    block.addEventListener('click', () => {
      layerBlocks.forEach(b => b.classList.remove('active'));
      block.classList.add('active');
      const layerKey = block.getAttribute('data-layer');
      if (layerDetailBox && LAYER_DESCRIPTIONS[layerKey]) {
        layerDetailBox.innerHTML = LAYER_DESCRIPTIONS[layerKey];
      }
    });
  });


  // =========================================================================
  // 6. PROCEDURAL SPECIMEN GENERATOR & GRAD-CAM SALIENCY RENDERER
  // =========================================================================
  function drawLeafSpecimen(specimenKey, showGradCam = false) {
    if (!canvasCtx || !leafCanvas) return;

    const width = leafCanvas.width;
    const height = leafCanvas.height;
    canvasCtx.clearRect(0, 0, width, height);

    // If custom user image was uploaded, draw it instead of procedural synthesis
    if (customUploadedImage) {
      canvasCtx.drawImage(customUploadedImage, 0, 0, width, height);
      if (showGradCam) {
        renderSimulatedGradCamOverlay(width, height, [{ x: 0.5, y: 0.5, r: 50 }]);
      }
      return;
    }

    const data = SPECIMEN_DATABASE[specimenKey];
    if (!data) return;

    canvasCtx.save();

    // 1. Draw Botanical Leaf Morphology
    drawLeafContour(canvasCtx, width, height, data);

    // 2. Draw Leaf Veins
    drawLeafVeins(canvasCtx, width, height, data);

    // 3. Draw Disease Pathological Lesions
    drawLesions(canvasCtx, width, height, data);

    canvasCtx.restore();

    // 4. If Grad-CAM toggle is on, render the Class Activation Heatmap
    if (showGradCam) {
      const lesionLocations = data.lesions || [];
      const pustuleLocations = data.pustules ? data.pustules.map(p => ({ x: p.x, y: p.y, r: 12 })) : [];
      const targets = [...lesionLocations, ...pustuleLocations];
      renderSimulatedGradCamOverlay(width, height, targets);
    }
  }

  function drawLeafContour(ctx, w, h, data) {
    const cx = w * 0.5;
    const cy = h * 0.52;

    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
    ctx.shadowBlur = 24;

    const grad = ctx.createRadialGradient(cx, cy, 30, cx, cy, w * 0.45);
    grad.addColorStop(0, data.baseColor);
    grad.addColorStop(1, adjustColor(data.baseColor, -35));
    ctx.fillStyle = grad;

    ctx.beginPath();
    if (data.leafType === 'lanceolate') {
      // Corn leaf: elongated, tapered blade
      ctx.moveTo(cx, h * 0.08);
      ctx.bezierCurveTo(cx + w * 0.18, h * 0.35, cx + w * 0.15, h * 0.70, cx + 8, h * 0.94);
      ctx.lineTo(cx - 8, h * 0.94);
      ctx.bezierCurveTo(cx - w * 0.15, h * 0.70, cx - w * 0.18, h * 0.35, cx, h * 0.08);
    } else {
      // Broad / Lobed leaf (Tomato, Potato, Bell Pepper, Apple)
      ctx.moveTo(cx, h * 0.12);
      ctx.bezierCurveTo(cx + w * 0.38, h * 0.22, cx + w * 0.42, h * 0.65, cx + 6, h * 0.90);
      ctx.lineTo(cx - 6, h * 0.90);
      ctx.bezierCurveTo(cx - w * 0.42, h * 0.65, cx - w * 0.38, h * 0.22, cx, h * 0.12);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawLeafVeins(ctx, w, h, data) {
    const cx = w * 0.5;
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';

    // Midrib
    ctx.beginPath();
    ctx.moveTo(cx, h * 0.13);
    ctx.quadraticCurveTo(cx + 2, h * 0.55, cx, h * 0.92);
    ctx.stroke();

    // Lateral secondary veins
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
    const veinCount = data.leafType === 'lanceolate' ? 14 : 7;
    for (let i = 1; i <= veinCount; i++) {
      const y = h * (0.18 + (i * 0.09));
      if (y > h * 0.85) break;

      // Right vein
      ctx.beginPath();
      ctx.moveTo(cx, y);
      ctx.quadraticCurveTo(cx + w * 0.15, y - 10, cx + w * 0.32, y - 20);
      ctx.stroke();

      // Left vein
      ctx.beginPath();
      ctx.moveTo(cx, y);
      ctx.quadraticCurveTo(cx - w * 0.15, y - 10, cx - w * 0.32, y - 20);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawLesions(ctx, w, h, data) {
    // 1. Draw Early Blight or Scab circular lesions with concentric rings
    if (data.lesions) {
      data.lesions.forEach(lesion => {
        const lx = lesion.x * w;
        const ly = lesion.y * h;

        // Chlorotic yellow halo
        if (lesion.halo) {
          const haloGrad = ctx.createRadialGradient(lx, ly, lesion.r * 0.7, lx, ly, lesion.halo);
          haloGrad.addColorStop(0, 'rgba(234, 179, 8, 0.65)');
          haloGrad.addColorStop(1, 'rgba(234, 179, 8, 0.0)');
          ctx.fillStyle = haloGrad;
          ctx.beginPath();
          ctx.arc(lx, ly, lesion.halo, 0, Math.PI * 2);
          ctx.fill();
        }

        // Necrotic core
        ctx.fillStyle = lesion.velvety ? '#3f3928' : (lesion.blotch ? '#281a13' : '#3e2415');
        ctx.beginPath();
        ctx.arc(lx, ly, lesion.r, 0, Math.PI * 2);
        ctx.fill();

        // Concentric target rings (Alternaria hallmark)
        if (lesion.rings > 0) {
          ctx.strokeStyle = '#1e110a';
          ctx.lineWidth = 2;
          for (let r = 1; r <= lesion.rings; r++) {
            ctx.beginPath();
            ctx.arc(lx, ly, (lesion.r / (lesion.rings + 1)) * r, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      });
    }

    // 2. Draw Corn Rust Pustules
    if (data.pustules) {
      ctx.save();
      data.pustules.forEach(p => {
        const px = p.x * w;
        const py = p.y * h;

        // Rust spot
        ctx.fillStyle = '#b45309';
        ctx.beginPath();
        ctx.ellipse(px, py, 6, 12, 0.1, 0, Math.PI * 2);
        ctx.fill();

        // Cinnamon powdery center
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.ellipse(px, py, 3, 7, 0.1, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }
  }

  // Simulated Grad-CAM Class Activation Map
  function renderSimulatedGradCamOverlay(w, h, targets) {
    if (!canvasCtx) return;

    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'source-over';

    // Darken background slightly to emphasize attention regions
    canvasCtx.fillStyle = 'rgba(4, 18, 48, 0.45)';
    canvasCtx.fillRect(0, 0, w, h);

    // If healthy leaf with no lesions, produce mild uniform attention across green leaf veins
    if (targets.length === 0) {
      const cx = w * 0.5;
      const cy = h * 0.5;
      const hGrad = canvasCtx.createRadialGradient(cx, cy, 20, cx, cy, w * 0.35);
      hGrad.addColorStop(0, 'rgba(59, 130, 246, 0.5)');
      hGrad.addColorStop(0.6, 'rgba(16, 185, 129, 0.3)');
      hGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      canvasCtx.fillStyle = hGrad;
      canvasCtx.beginPath();
      canvasCtx.arc(cx, cy, w * 0.35, 0, Math.PI * 2);
      canvasCtx.fill();
      canvasCtx.restore();
      return;
    }

    // Render multi-level Jet heatmap on lesion clusters
    targets.forEach(t => {
      const tx = (t.x || 0.5) * w;
      const ty = (t.y || 0.5) * h;
      const radius = Math.max(t.r || 35, 30) * 1.7;

      const grad = canvasCtx.createRadialGradient(tx, ty, 5, tx, ty, radius);
      grad.addColorStop(0.0, 'rgba(239, 68, 68, 0.85)');    // Red core (Max CNN activation)
      grad.addColorStop(0.25, 'rgba(245, 158, 11, 0.75)'); // Orange
      grad.addColorStop(0.50, 'rgba(234, 179, 8, 0.60)');  // Yellow
      grad.addColorStop(0.75, 'rgba(16, 185, 129, 0.35)'); // Green
      grad.addColorStop(1.0, 'rgba(59, 130, 246, 0.0)');   // Transparent blue outer
      canvasCtx.fillStyle = grad;

      canvasCtx.beginPath();
      canvasCtx.arc(tx, ty, radius, 0, Math.PI * 2);
      canvasCtx.fill();
    });

    canvasCtx.restore();
  }

  function adjustColor(hex, lum) {
    hex = String(hex).replace(/[^0-9a-f]/gi, '');
    if (hex.length < 6) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    lum = lum || 0;
    let rgb = "#", c, i;
    for (i = 0; i < 3; i++) {
      c = parseInt(hex.substr(i*2,2), 16);
      c = Math.round(Math.min(Math.max(0, c + lum), 255)).toString(16);
      rgb += ("00"+c).substr(c.length);
    }
    return rgb;
  }


  // =========================================================================
  // 7. DIAGNOSIS CARD & TELEMETRY UPDATER
  // =========================================================================
  function updateDiagnosisCard(specimenKey) {
    const data = SPECIMEN_DATABASE[specimenKey];
    if (!data) return;

    if (canvasTag) canvasTag.textContent = `Specimen: ${data.crop}`;
    if (diagName) diagName.textContent = data.disease;
    if (diagPathogen) diagPathogen.innerHTML = `Pathogen: <em>${data.pathogen}</em>`;
    if (diagTypePill) {
      diagTypePill.textContent = data.type;
      diagTypePill.className = `pathogen-type-pill ${data.typeBadgeClass}`;
    }
    if (diagConfPill) {
      diagConfPill.textContent = `${data.confidence}% Confidence`;
    }
    if (diagBanner) {
      diagBanner.classList.toggle('healthy', data.typeBadgeClass === 'success');
    }
    if (badgeInferenceStatus) {
      badgeInferenceStatus.textContent = `Inference Active • ${data.latency}ms`;
    }

    // Update Top-3 Probabilities
    if (probBarsGroup) {
      probBarsGroup.innerHTML = '';
      data.top3.forEach(item => {
        const itemEl = document.createElement('div');
        itemEl.className = 'prob-item';
        itemEl.innerHTML = `
          <div class="prob-header">
            <span>${item.name}</span>
            <span class="prob-pct">${item.prob}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${item.prob}%;"></div>
          </div>
        `;
        probBarsGroup.appendChild(itemEl);
      });
    }

    // Update Prescriptions
    if (rxOrganic) rxOrganic.textContent = data.organicRx;
    if (rxChemical) rxChemical.textContent = data.chemicalRx;
    if (rxPrevention) rxPrevention.innerHTML = `<strong>Cultural Practice:</strong> ${data.prevention}`;

    // Render Canvas Specimen
    drawLeafSpecimen(specimenKey, isGradCamActive);
  }


  // =========================================================================
  // 8. INTERACTIVE LAB EVENT LISTENERS
  // =========================================================================
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sampleBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      customUploadedImage = null; // Clear custom upload if selecting sample
      currentSampleId = btn.getAttribute('data-sample');
      updateDiagnosisCard(currentSampleId);
    });
  });

  if (btnViewRaw) {
    btnViewRaw.addEventListener('click', () => {
      btnViewRaw.classList.add('active');
      btnViewRaw.setAttribute('aria-pressed', 'true');
      btnViewGradcam.classList.remove('active');
      btnViewGradcam.setAttribute('aria-pressed', 'false');
      isGradCamActive = false;
      drawLeafSpecimen(currentSampleId, false);
    });
  }

  if (btnViewGradcam) {
    btnViewGradcam.addEventListener('click', () => {
      btnViewGradcam.classList.add('active');
      btnViewGradcam.setAttribute('aria-pressed', 'true');
      btnViewRaw.classList.remove('active');
      btnViewRaw.setAttribute('aria-pressed', 'false');
      isGradCamActive = true;
      drawLeafSpecimen(currentSampleId, true);
    });
  }

  // Drag & Drop / File Upload Handler
  if (dropZone && leafFileInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files[0]) handleUploadedFile(files[0]);
    });

    leafFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleUploadedFile(e.target.files[0]);
      }
    });
  }

  function handleUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (PNG, JPG, WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        customUploadedImage = img;
        sampleBtns.forEach(b => b.classList.remove('active'));

        // Run diagnosis simulation on uploaded image
        if (canvasTag) canvasTag.textContent = `Uploaded: ${file.name}`;
        if (diagName) diagName.textContent = "Custom Leaf Specimen Analyzed";
        if (diagPathogen) diagPathogen.innerHTML = "Diagnosis: <em>Simulated CNN Diagnostic Scan</em>";
        if (diagConfPill) diagConfPill.textContent = "96.4% Confidence";
        if (diagTypePill) {
          diagTypePill.textContent = "Detected Pathology";
          diagTypePill.className = "pathogen-type-pill alert";
        }

        // Draw custom image onto canvas
        drawLeafSpecimen(currentSampleId, isGradCamActive);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }


  // =========================================================================
  // 9. DATA VISUALIZATION: CANVAS CHART RENDERERS
  // =========================================================================

  // Crop Distribution Bar Chart (Slide 2)
  function renderCropDistributionChart() {
    const canvas = document.getElementById('crop-dist-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const crops = [
      { name: 'Tomato', count: 18160, color: '#ef4444' },
      { name: 'Potato', count: 4300, color: '#f59e0b' },
      { name: 'Corn', count: 7710, color: '#eab308' },
      { name: 'Apple', count: 6340, color: '#10b981' },
      { name: 'Grape', count: 8140, color: '#8b5cf6' },
      { name: 'Pepper', count: 4950, color: '#06b6d4' }
    ];

    const maxVal = 20000;
    const barWidth = 38;
    const gap = 24;
    const startX = 40;
    const baselineY = h - 35;
    const chartHeight = h - 60;

    // Draw Y-Axis grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let step = 0; step <= maxVal; step += 5000) {
      const y = baselineY - (step / maxVal) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(35, y);
      ctx.lineTo(w - 15, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`${step / 1000}k`, 8, y + 3);
    }

    // Draw Bars
    crops.forEach((crop, idx) => {
      const x = startX + idx * (barWidth + gap);
      const barH = (crop.count / maxVal) * chartHeight;
      const y = baselineY - barH;

      // Bar
      ctx.fillStyle = crop.color;
      ctx.fillRect(x, y, barWidth, barH);

      // Value label on top
      ctx.fillStyle = '#f1f5f9';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${(crop.count / 1000).toFixed(1)}k`, x + barWidth / 2, y - 6);

      // Crop label on bottom
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px "Plus Jakarta Sans", sans-serif';
      ctx.fillText(crop.name, x + barWidth / 2, baselineY + 18);
    });
  }

  // Training & Validation Curves Chart (Slide 6)
  function renderTrainingCurvesChart() {
    const canvas = document.getElementById('training-curves-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const epochs = 25;
    const trainAcc = [
      0.65, 0.74, 0.81, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96,
      0.965, 0.97, 0.973, 0.977, 0.980, 0.982, 0.985, 0.987, 0.989, 0.990,
      0.991, 0.992, 0.993, 0.994, 0.995
    ];
    const valAcc = [
      0.62, 0.71, 0.79, 0.83, 0.87, 0.89, 0.91, 0.925, 0.938, 0.948,
      0.954, 0.960, 0.965, 0.969, 0.972, 0.975, 0.978, 0.981, 0.982, 0.983,
      0.984, 0.984, 0.984, 0.985, 0.984
    ];

    const padding = { left: 45, right: 20, top: 25, bottom: 35 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    [0.6, 0.7, 0.8, 0.9, 1.0].forEach(val => {
      const y = padding.top + chartH - ((val - 0.5) / 0.5) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${(val * 100).toFixed(0)}%`, padding.left - 6, y + 3);
    });

    // Draw Train Line (Blue)
    drawLine(ctx, trainAcc, '#3b82f6', padding, chartW, chartH, epochs);

    // Draw Val Line (Green)
    drawLine(ctx, valAcc, '#10b981', padding, chartW, chartH, epochs);

    // X-Axis Epoch Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    [1, 5, 10, 15, 20, 25].forEach(ep => {
      const x = padding.left + ((ep - 1) / (epochs - 1)) * chartW;
      ctx.fillText(`E${ep}`, x, h - 12);
    });
  }

  function drawLine(ctx, data, color, pad, cw, ch, total) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    data.forEach((val, idx) => {
      const x = pad.left + (idx / (total - 1)) * cw;
      const y = pad.top + ch - ((val - 0.5) / 0.5) * ch;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  // Interactive Confusion Matrix (Slide 7)
  function renderConfusionMatrix() {
    const wrapper = document.getElementById('cm-interactive-wrapper');
    if (!wrapper || wrapper.innerHTML.trim() !== '') return;

    const classes = ['Tom_EB', 'Tom_LB', 'Pot_EB', 'Pot_LB', 'Corn_Rust', 'App_Scab'];
    // Representative Confusion Matrix counts
    const matrix = [
      [245,   8,   2,   0,   0,   0],
      [  7, 238,   1,   3,   0,   0],
      [  3,   1, 210,   5,   0,   0],
      [  0,   4,   4, 212,   0,   0],
      [  0,   0,   0,   0, 250,   1],
      [  0,   0,   0,   0,   0, 220]
    ];

    let html = '<table class="cm-table"><thead><tr><th>True \\ Pred</th>';
    classes.forEach(c => { html += `<th>${c}</th>`; });
    html += '</tr></thead><tbody>';

    matrix.forEach((row, rIdx) => {
      html += `<tr><th>${classes[rIdx]}</th>`;
      row.forEach((val, cIdx) => {
        let cellClass = 'zero';
        if (rIdx === cIdx) cellClass = 'high';
        else if (val > 3) cellClass = 'low';
        else if (val > 0) cellClass = 'med';

        html += `<td class="cm-cell ${cellClass}" title="True: ${classes[rIdx]}, Pred: ${classes[cIdx]} (${val} samples)">${val}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';

    wrapper.innerHTML = html;
  }

  // =========================================================================
  // 10. CODE DOWNLOAD & CLIPBOARD UTILITIES
  // =========================================================================
  if (btnCopyScript) {
    btnCopyScript.addEventListener('click', () => {
      const snippet = `# Run plant leaf CNN training
python train_plant_cnn.py --epochs 25 --batch-size 32 --model-type custom_cnn`;
      navigator.clipboard.writeText(snippet).then(() => {
        const originalText = btnCopyScript.textContent;
        btnCopyScript.textContent = 'Copied!';
        btnCopyScript.style.background = 'rgba(16, 185, 129, 0.3)';
        setTimeout(() => {
          btnCopyScript.textContent = originalText;
          btnCopyScript.style.background = '';
        }, 2000);
      });
    });
  }

  // Initialize First Slide and Default Lab Specimen
  goToSlide(0);
  updateDiagnosisCard(currentSampleId);

});
