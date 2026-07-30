import { APIClient } from "./api.js";
import { LineChart } from "./charts.js";
import { MotifTrack } from "./motifs.js";
import igv from "../vendor/igv.esm.min.js";

const api = new APIClient();
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const escapeHTML = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character],
);
const manuscriptPlotEffects = (rawEffects, baseline) => {
  const finiteBaseline = baseline
    .map((value) => Math.abs(Number(value)))
    .filter(Number.isFinite);
  const denominator = finiteBaseline.length ? Math.max(...finiteBaseline) : 0;
  const applied = denominator > 0;
  return {
    values: rawEffects.map((value) => {
      const absolute = Math.abs(Number(value));
      return applied ? (absolute / denominator) * 100 : absolute;
    }),
    metadata: {
      id: "absolute_percent_of_max_predicted_output",
      formula: "100 × |raw deletion effect| ÷ max(|baseline prediction|)",
      denominator,
      applied,
      unit: applied ? "%" : "absolute raw effect",
      label: applied
        ? "Absolute deletion effect (% max predicted accessibility)"
        : "Absolute deletion effect",
    },
  };
};

class ModelStudioApp {
  constructor() {
    this.state = {
      capabilities: null,
      sourceKind: "region",
      sequence: null,
      model: null,
      profileModel: null,
      presetId: "custom",
      jobs: [],
      selectedJobId: null,
      result: null,
      annotationSummary: null,
      annotationFilterTimer: null,
      pollTimer: null,
      startedAt: null,
      igvBrowser: null,
      igvFrame: null,
      igvTrack: null,
      igvTrackCoverage: null,
      igvTrackTimer: null,
      igvTrackRequestId: 0,
    };
    this.baselineChart = new LineChart($("#baselineChart"), {
      color: "#007181",
      tooltip: $("#chartTooltip"),
    });
    this.deletionChart = new LineChart($("#deletionChart"), {
      color: "#c33a35",
      zeroLine: true,
    });
    this.motifTrack = new MotifTrack($("#motifCanvas"));
  }

  async initialize() {
    this.bindEvents();
    this.state.capabilities = await api.get("/api/capabilities");
    this.renderCapabilities();
    this.renderAnnotationMode();
    this.renderAnnotationThresholds();
    await this.initializeGenomeBrowser();
    await this.refreshModel();
    await this.refreshJobs();
    this.state.pollTimer = window.setInterval(() => this.refreshJobs(), 1200);
  }

  bindEvents() {
    $$("#sourceTabs button").forEach((button) => {
      button.addEventListener("click", () => this.selectSource(button.dataset.source));
    });
    $("#analysisPreset").addEventListener("change", () => {
      this.applyPreset($("#analysisPreset").value);
    });
    $("#architecture").addEventListener("change", () => this.renderAdapterOptions());
    $("#device").addEventListener("change", () => {
      this.applyDeviceBatchRecommendation();
    });
    $("#fastaFile").addEventListener("change", (event) => this.readFasta(event));
    $("#checkpointFile").addEventListener("change", (event) => this.uploadCheckpoint(event));
    $("#motifFile").addEventListener("change", (event) => {
      this.uploadTable(event, $("#motifPath"), "motif");
    });
    $("#rnaFile").addEventListener("change", (event) => {
      this.uploadTable(event, $("#rnaPath"), "RNA");
    });
    $("#validateSequence").addEventListener("click", () => this.validateSequence());
    $("#loadModel").addEventListener("click", () => this.loadModel());
    $("#runInference").addEventListener("click", () => this.submitRun());
    $("#runIgvRegion").addEventListener("click", () => this.runIgvRegion());
    $("#loadAnnotations").addEventListener("click", () => this.loadAnnotations());
    $("#refreshJobs").addEventListener("click", () => this.refreshJobs());
    $("#resetZoom").addEventListener("click", () => {
      this.baselineChart.resetZoom();
      this.deletionChart.resetZoom();
    });
    $("#downloadJson").addEventListener("click", () => this.download("result.json"));
    $("#downloadCsv").addEventListener("click", () => this.download("effects.csv"));
    $$("[name='annotationMode']").forEach((input) => {
      input.addEventListener("change", () => {
        this.renderAnnotationMode();
        this.scheduleAnnotationFilter();
      });
    });
    ["#logfcThreshold", "#expressionThreshold", "#padjThreshold", "#effectThreshold"].forEach((selector) => {
      $(selector).addEventListener("input", () => {
        this.renderAnnotationThresholds();
        this.scheduleAnnotationFilter();
      });
    });
    $("#tfSelector").addEventListener("change", () => this.scheduleAnnotationFilter());
  }

  parseModelRegion(value) {
    const match = String(value).replaceAll(",", "").trim().match(
      /^([^:]+):(\d+)-(\d+)$/,
    );
    if (!match) throw new Error("Region must use chromosome:start-end syntax");
    const start = Number(match[2]);
    const end = Number(match[3]);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      throw new Error("Region end must be greater than start");
    }
    return { chr: match[1], start, end };
  }

  modelRegionToIgvLocus(value) {
    const region = this.parseModelRegion(value);
    return `${region.chr}:${region.start + 1}-${region.end}`;
  }

  formatIgvRegion(region) {
    return `${region.chr}:${(region.start + 1).toLocaleString()}–`
      + `${region.end.toLocaleString()}`;
  }

  setIgvStatus(state, message) {
    $("#igvStatus").className = `igv-status ${state}`;
    $("#igvStatus").textContent = message;
  }

  normalizeIgvFrame(frame) {
    if (!frame?.chr) return null;
    const start = Math.max(0, Math.floor(Number(frame.start)));
    const end = Math.max(start + 1, Math.ceil(Number(frame.end)));
    return { chr: frame.chr, start, end };
  }

  async initializeGenomeBrowser() {
    const presetId = $("#analysisPreset").value;
    const initialRegion = this.parseModelRegion($("#region").value);
    this.setIgvStatus(
      "loading",
      "Loading the local mm10 reference and observed ATAC track…",
    );
    try {
      const browser = await igv.createBrowser($("#igvBrowser"), {
        reference: {
          id: "mm10",
          name: "Mouse mm10",
          format: "chromsizes",
          fastaURL: `/api/genome/mm10.chrom.sizes?preset_id=${encodeURIComponent(presetId)}`,
        },
        locus: this.modelRegionToIgvLocus($("#region").value),
        minimumBases: 32,
        showIdeogram: true,
        showSVGButton: false,
        showTrackLabelButton: false,
        showCenterGuideButton: false,
        showCursorTrackingGuideButton: false,
        showSampleNameButton: false,
        tracks: [],
      });
      this.state.igvBrowser = browser;
      $("#runIgvRegion").disabled = false;
      browser.on("locuschange", (frames) => {
        const frame = Array.isArray(frames) ? frames[0] : frames;
        this.handleIgvLocusChange(frame);
      });
      browser.on("trackdragend", () => {
        this.scheduleObservedTrack(this.state.igvFrame, true);
      });
      this.handleIgvLocusChange(initialRegion);
      window.clearTimeout(this.state.igvTrackTimer);
      await this.loadObservedTrack(initialRegion);
    } catch (error) {
      $("#runIgvRegion").disabled = true;
      this.setIgvStatus("error", `Genome browser unavailable: ${error.message}`);
    }
  }

  handleIgvLocusChange(frame) {
    const normalized = this.normalizeIgvFrame(frame);
    if (!normalized) return;
    this.state.igvFrame = normalized;
    $("#igvLocus").textContent = `${this.formatIgvRegion(normalized)} · 1-based`;
    this.scheduleObservedTrack(normalized);
  }

  trackCoverageContains(region) {
    const coverage = this.state.igvTrackCoverage;
    if (!coverage || coverage.chr !== region.chr) return false;
    const margin = Math.min(
      20_000,
      Math.floor((coverage.end - coverage.start) * 0.1),
    );
    return (
      region.start >= coverage.start + margin
      && region.end <= coverage.end - margin
    );
  }

  scheduleObservedTrack(region, immediate = false) {
    if (!region || this.trackCoverageContains(region)) return;
    window.clearTimeout(this.state.igvTrackTimer);
    this.state.igvTrackTimer = window.setTimeout(
      () => this.loadObservedTrack(region),
      immediate ? 0 : 240,
    );
  }

  observedTrackCoverage(region) {
    const viewSpan = Math.max(1, region.end - region.start);
    const span = Math.min(2_000_000, Math.max(100_000, viewSpan * 6));
    const center = (region.start + region.end) / 2;
    const start = Math.max(0, Math.floor(center - span / 2));
    return {
      chr: region.chr,
      start,
      end: Math.ceil(start + span),
    };
  }

  async loadObservedTrack(region) {
    const browser = this.state.igvBrowser;
    if (!browser || !region) return;
    const preset = this.selectedPreset();
    if (!preset?.observed_track) {
      if (this.state.igvTrack) browser.removeTrack(this.state.igvTrack);
      this.state.igvTrack = null;
      this.state.igvTrackCoverage = null;
      this.setIgvStatus(
        "success",
        "Reference ready. This custom workflow has no observed ATAC track.",
      );
      return;
    }
    const requestId = ++this.state.igvTrackRequestId;
    const coverage = this.observedTrackCoverage(region);
    const parameters = new URLSearchParams({
      preset_id: preset.id,
      chr: coverage.chr,
      start: String(coverage.start),
      end: String(coverage.end),
      max_points: "50000",
    });
    this.setIgvStatus("loading", "Loading observed ATAC for this view…");
    try {
      const nextTrack = await browser.loadTrack({
        id: `observed-atac-${requestId}`,
        name: "Observed ATAC · SCCApos",
        type: "wig",
        format: "bedgraph",
        url: `/api/tracks/observed-atac.bedgraph?${parameters}`,
        indexed: false,
        color: "#007181",
        autoscale: true,
        height: 72,
        minHeight: 54,
        maxHeight: 120,
      });
      if (!nextTrack) throw new Error("IGV did not create the observed track");
      if (requestId !== this.state.igvTrackRequestId) {
        browser.removeTrack(nextTrack);
        return;
      }
      if (this.state.igvTrack) browser.removeTrack(this.state.igvTrack);
      this.state.igvTrack = nextTrack;
      this.state.igvTrackCoverage = coverage;
      this.setIgvStatus(
        "success",
        "Observed ATAC ready · Shift-drag the ruler for an ROI, or run the current view.",
      );
    } catch (error) {
      if (requestId !== this.state.igvTrackRequestId) return;
      this.setIgvStatus("error", `Observed ATAC could not load: ${error.message}`);
    }
  }

  async runIgvRegion() {
    const browser = this.state.igvBrowser;
    const button = $("#runIgvRegion");
    if (!browser || !this.state.igvFrame) return;
    button.disabled = true;
    try {
      const frame = this.state.igvFrame;
      const rois = await browser.getUserDefinedROIs();
      const visibleRois = (rois || []).filter((roi) => (
        roi.chr === frame.chr
        && Number(roi.end) > frame.start
        && Number(roi.start) < frame.end
      ));
      const selected = visibleRois.at(-1);
      const source = selected || frame;
      let start = Math.max(0, Math.floor(Number(source.start)));
      let end = Math.ceil(Number(source.end));
      if (end - start < 32) {
        const center = (start + end) / 2;
        start = Math.max(0, Math.floor(center - 16));
        end = start + 32;
      }
      const span = end - start;
      if (span > 2000) {
        throw new Error(
          "Select or zoom to 2,000 bp or less before running; the model still uses its fixed 4,096-bp context.",
        );
      }
      const region = { chr: source.chr, start, end };
      $("#region").value = `${region.chr}:${region.start}-${region.end}`;
      $("#displayBp").value = String(span);
      this.state.sequence = null;
      this.setIgvStatus(
        "loading",
        `Preparing ${selected ? "selected ROI" : "current view"} `
          + `${this.formatIgvRegion(region)}…`,
      );
      const sequence = await this.validateSequence();
      if (!sequence) throw new Error($("#sequenceStatus").textContent);
      if (!this.state.model) {
        const model = await this.loadModel();
        if (!model) throw new Error($("#modelStatus").textContent);
      }
      const job = await this.submitRun();
      if (!job) throw new Error($("#runError").textContent || "Run was not submitted");
      this.setIgvStatus(
        "success",
        `${selected ? "Selected ROI" : "Current view"} queued · `
          + `${this.formatIgvRegion(region)}.`,
      );
    } catch (error) {
      this.setIgvStatus("error", error.message);
    } finally {
      button.disabled = !this.state.igvBrowser;
    }
  }

  async syncGenomeBrowserToRegion(regionValue) {
    if (!this.state.igvBrowser || !regionValue) return;
    this.state.igvTrackCoverage = null;
    await this.state.igvBrowser.search(this.modelRegionToIgvLocus(regionValue));
    window.clearTimeout(this.state.igvTrackTimer);
    await this.loadObservedTrack(this.state.igvFrame);
  }

  renderCapabilities() {
    const { capabilities } = this.state;
    const {
      devices,
      defaults,
      architectures,
      analysis_presets: presets,
    } = capabilities;
    $("#topDevice").textContent = `Device · ${devices.recommended}`;
    $("#architecture").innerHTML = architectures
      .map((item) => `<option value="${item.id}">${item.label}</option>`)
      .join("");
    $("#analysisPreset").innerHTML = presets
      .map((item) => `<option value="${item.id}">${escapeHTML(item.label)}</option>`)
      .join("");
    $("#analysisPreset").value = defaults.preset_id || "custom";
    this.applyPreset($("#analysisPreset").value);
    this.renderAdapterOptions();
  }

  selectedPreset() {
    return this.state.capabilities.analysis_presets.find(
      (item) => item.id === $("#analysisPreset").value,
    );
  }

  applyPreset(presetId) {
    const preset = this.state.capabilities.analysis_presets.find(
      (item) => item.id === presetId,
    );
    if (!preset) return;
    this.state.presetId = preset.id;
    $("#presetDescription").textContent = preset.description;
    const fixed = preset.id !== "custom";
    const provenance = preset.provenance
      ? `Figure workflow · ${preset.provenance.analysis}`
      : "Custom workflow · every model and filter choice is user controlled";
    this.setValidation($("#presetProvenance"), fixed ? "success" : "neutral", provenance);
    $$(".custom-only").forEach((element) => {
      element.classList.toggle("hidden", fixed);
    });
    $("#sourceTabs").classList.toggle("hidden", fixed);
    if (fixed) this.selectSource("region");
    if (preset.source) {
      $("#region").value = preset.source.region;
      $("#genomeZarr").value = preset.source.genome_zarr;
      $("#contextBp").value = preset.context_bp;
      $("#displayBp").value = preset.display_bp;
    }
    if (preset.analysis_model) {
      $("#architecture").value = preset.analysis_model.architecture_id;
      this.renderAdapterOptions();
      $("#checkpointPath").value = preset.analysis_model.checkpoint;
      Object.entries(preset.analysis_model.options || {}).forEach(([name, value]) => {
        const input = document.querySelector(`[data-adapter-option="${name}"]`);
        if (input) input.value = String(value);
      });
      $("#profileCheckpointPath").value = preset.profile_model?.checkpoint || "";
    }
    if (preset.deletion) {
      $("#deletionSize").value = preset.deletion.deletion_size;
      $("#stride").value = preset.deletion.stride;
      $("#batchSize").value = preset.deletion.batch_size;
      $("#smoothing").value = preset.deletion.smoothing_window;
      $("#scanScope").value = preset.deletion.scan_scope || "context";
      const perturbation = document.querySelector(
        `[name="perturbation"][value="${preset.deletion.perturbation}"]`,
      );
      if (perturbation) perturbation.checked = true;
      this.applyDeviceBatchRecommendation();
    }
    if (preset.annotations) {
      $("#motifPath").value = preset.annotations.motif_path || "";
      $("#rnaPath").value = preset.annotations.rna_path || "";
      $("#rnaPath2").value = preset.annotations.rna_path_2 || "";
      $("#geneListPath").value = preset.annotations.gene_list_path || "";
      $("#effectThreshold").value = preset.annotations.min_effect_pct ?? 10;
      const mode = document.querySelector(
        `[name="annotationMode"][value="${preset.annotations.default_mode || "expression"}"]`,
      );
      if (mode) mode.checked = true;
    }
    [
      "#region",
      "#genomeZarr",
      "#contextBp",
      "#displayBp",
      "#architecture",
      "#checkpointPath",
      "#profileCheckpointPath",
    ].forEach((selector) => {
      $(selector).disabled = fixed;
    });
    $$("[data-adapter-option]").forEach((input) => { input.disabled = fixed; });
    this.state.sequence = null;
    this.state.model = null;
    this.state.profileModel = null;
    this.setValidation($("#sequenceStatus"), "neutral", "Validate this workflow's model context");
    this.setValidation($("#modelStatus"), "neutral", "Load this workflow's checkpoint role(s)");
    this.renderAnnotationMode();
    this.renderAnnotationThresholds();
    this.renderModel();
    this.updateRunEnabled();
    if (this.state.igvBrowser && preset.source?.region) {
      this.syncGenomeBrowserToRegion(preset.source.region).catch((error) => {
        this.setIgvStatus("error", `Genome browser could not move: ${error.message}`);
      });
    }
  }

  applyDeviceBatchRecommendation() {
    if (!this.state.capabilities || $("#analysisPreset").value === "custom") return;
    const requested = $("#device").value;
    const effective = requested === "auto"
      ? this.state.capabilities.devices.recommended
      : requested;
    let batchSize = 4;
    let label = "CPU";
    if (String(effective).startsWith("cuda")) {
      batchSize = 32;
      label = "CUDA";
    } else if (String(effective).startsWith("mps")) {
      batchSize = 8;
      label = "Apple MPS";
    }
    $("#batchSize").value = String(batchSize);
    $("#batchHelp").textContent =
      `${label} safe default: batch ${batchSize}. Increase it if memory allows; `
      + "use batch 1 for the legacy one-deletion-at-a-time numerical path.";
  }

  architectureDescriptor() {
    const architectureId = $("#architecture").value;
    return this.state.capabilities.architectures.find((item) => item.id === architectureId);
  }

  renderAdapterOptions() {
    const descriptor = this.architectureDescriptor();
    if (!descriptor) return;
    $("#adapterOptions").innerHTML = (descriptor.option_schema || [])
      .map((option) => {
        if (option.type === "select") {
          return `<label><span>${option.label}</span><select data-adapter-option="${option.name}">${option.choices
            .map((choice) => `<option value="${choice}" ${choice === option.default ? "selected" : ""}>${choice}</option>`)
            .join("")}</select></label>`;
        }
        return `<label><span>${option.label}${option.required ? " *" : ""}</span><input data-adapter-option="${option.name}" value="${option.default || ""}" spellcheck="false"></label>`;
      })
      .join("");
  }

  selectSource(kind) {
    this.state.sourceKind = kind;
    $$("#sourceTabs button").forEach((button) => {
      button.classList.toggle("active", button.dataset.source === kind);
    });
    $$("[data-source-panel]").forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.sourcePanel !== kind);
    });
    this.state.sequence = null;
    this.setValidation($("#sequenceStatus"), "neutral", "Sequence not validated");
    this.updateRunEnabled();
  }

  async readFasta(event) {
    const [file] = event.target.files;
    if (!file) return;
    $("#sequenceText").value = await file.text();
    this.state.sourceKind = "fasta";
  }

  async uploadCheckpoint(event) {
    const [file] = event.target.files;
    if (!file) return;
    this.setValidation($("#modelStatus"), "loading", `Uploading ${file.name}…`);
    try {
      const payload = await api.uploadCheckpoint(file);
      $("#checkpointPath").value = payload.path;
      this.setValidation($("#modelStatus"), "neutral", "Checkpoint uploaded; load it to validate");
    } catch (error) {
      this.setValidation($("#modelStatus"), "error", error.message);
    }
  }

  async uploadTable(event, pathInput, label) {
    const [file] = event.target.files;
    if (!file) return;
    this.setValidation(
      $("#annotationStatus"),
      "loading",
      `Uploading ${label} table ${file.name}…`,
    );
    try {
      const payload = await api.uploadTable(file);
      pathInput.value = payload.path;
      this.setValidation(
        $("#annotationStatus"),
        "neutral",
        `${label} table uploaded; load annotations to validate`,
      );
    } catch (error) {
      this.setValidation($("#annotationStatus"), "error", error.message);
    }
  }

  sequencePayload() {
    if (this.state.sourceKind === "region") {
      return {
        source: {
          kind: "region",
          region: $("#region").value.trim(),
          genome_zarr: $("#genomeZarr").value.trim(),
        },
        context_bp: Number($("#contextBp").value),
        display_bp: Number($("#displayBp").value),
      };
    }
    return {
      source: {
        kind: this.state.sourceKind,
        sequence: $("#sequenceText").value,
        label: this.state.sourceKind === "fasta" ? ($("#fastaFile").files[0]?.name || "FASTA") : "Pasted sequence",
      },
    };
  }

  async validateSequence() {
    this.setValidation($("#sequenceStatus"), "loading", "Validating sequence…");
    try {
      const payload = await api.post("/api/sequence/inspect", this.sequencePayload());
      this.state.sequence = payload.sequence;
      this.setValidation(
        $("#sequenceStatus"),
        "success",
        `Valid · ${payload.sequence.length.toLocaleString()} bp · ${(payload.sequence.gc_fraction * 100).toFixed(1)}% GC`,
      );
      this.renderInputDetails();
    } catch (error) {
      this.state.sequence = null;
      this.setValidation($("#sequenceStatus"), "error", error.message);
    }
    this.updateRunEnabled();
    return this.state.sequence;
  }

  adapterOptions() {
    return Object.fromEntries(
      $$("[data-adapter-option]").map((input) => [input.dataset.adapterOption, input.value.trim()]),
    );
  }

  async loadModel() {
    this.setValidation($("#modelStatus"), "loading", "Loading and validating checkpoint…");
    $("#loadModel").disabled = true;
    try {
      const payload = await api.post("/api/model/load", {
        preset_id: $("#analysisPreset").value,
        architecture_id: $("#architecture").value,
        checkpoint: $("#checkpointPath").value.trim(),
        profile_architecture_id: $("#architecture").value,
        profile_checkpoint: $("#profileCheckpointPath").value.trim() || null,
        device: $("#device").value,
        options: this.adapterOptions(),
        profile_options: this.adapterOptions(),
      });
      this.state.model = payload.model;
      this.state.profileModel = payload.profile_model;
      this.setValidation(
        $("#modelStatus"),
        "success",
        this.state.profileModel
          ? `Loaded deletion + profile checkpoints on ${payload.model.device}`
          : `Loaded on ${payload.model.device} · ${payload.model.output_heads.join(", ")}`,
      );
      this.renderModel();
    } catch (error) {
      this.state.model = null;
      this.state.profileModel = null;
      this.setValidation($("#modelStatus"), "error", error.message);
      this.renderModel();
    } finally {
      $("#loadModel").disabled = false;
      this.updateRunEnabled();
    }
    return this.state.model;
  }

  async refreshModel() {
    const payload = await api.get("/api/model");
    this.state.model = payload.model;
    this.state.profileModel = payload.profile_model;
    if (this.state.model) {
      $("#architecture").value = this.state.model.architecture_id;
      this.renderAdapterOptions();
      $("#checkpointPath").value = this.state.model.checkpoint;
      $("#device").value = this.state.model.device.startsWith("cuda")
        ? "cuda"
        : this.state.model.device;
      Object.entries(this.state.model.details || {}).forEach(([name, value]) => {
        const input = document.querySelector(`[data-adapter-option="${name}"]`);
        if (input && typeof value !== "object") input.value = String(value);
      });
      this.setValidation(
        $("#modelStatus"),
        "success",
        this.state.profileModel
          ? `Loaded deletion + profile checkpoints on ${this.state.model.device}`
          : `Loaded on ${this.state.model.device} · ${this.state.model.output_heads.join(", ")}`,
      );
    }
    this.renderModel();
  }

  renderModel() {
    const model = this.state.model;
    if (!model) {
      $("#modelSummary").innerHTML = "<strong>Model summary</strong><span>Load a checkpoint to begin.</span>";
      $("#modelDetails").innerHTML = this.detailRows([
        ["Architecture", "—"], ["Checkpoint", "—"], ["Device", "—"], ["Output heads", "—"],
      ]);
      $("#outputHead").disabled = true;
      return;
    }
    const profile = this.state.profileModel;
    $("#modelSummary").innerHTML = `
      <strong>Ready</strong>
      <span>Differential deletion${profile ? " + accessibility profile" : ""}</span>
      <span>${escapeHTML(model.device)}</span>`;
    $("#modelDetails").innerHTML = this.detailRows([
      ["Deletion model", model.checkpoint.split("/").pop()],
      ["Profile model", profile ? profile.checkpoint.split("/").pop() : "same / not configured"],
      ["Device", model.device],
      ["Output heads", model.output_heads.join(", ")],
    ]);
    $("#outputHead").innerHTML = model.output_heads
      .map((head) => `<option value="${head}">${head[0].toUpperCase()}${head.slice(1)}</option>`)
      .join("");
    $("#outputHead").disabled = false;
    const presetHead = this.selectedPreset()?.analysis_model?.output_head;
    if (
      presetHead
      && Array.from($("#outputHead").options).some((option) => option.value === presetHead)
    ) {
      $("#outputHead").value = presetHead;
    }
    $("#outputHead").disabled = $("#analysisPreset").value !== "custom";
    this.updateValidationList();
  }

  renderInputDetails(sequence = this.state.sequence) {
    $("#inputDetails").innerHTML = sequence
      ? this.detailRows([
          ["Source", sequence.label],
          ["Length", `${sequence.length.toLocaleString()} bp`],
          ["GC content", `${(sequence.gc_fraction * 100).toFixed(1)}%`],
          ["Scoring core", sequence.regions?.core || "full input"],
          ["Display", sequence.regions?.display || "full input"],
        ])
      : this.detailRows([
          ["Source", "—"],
          ["Length", "—"],
          ["GC content", "—"],
          ["Scoring core", "—"],
          ["Display", "—"],
        ]);
    this.updateValidationList();
  }

  detailRows(rows) {
    return rows
      .map(([term, value]) => `<div><dt>${escapeHTML(term)}</dt><dd>${escapeHTML(value)}</dd></div>`)
      .join("");
  }

  setValidation(element, state, message) {
    element.className = `validation ${state}`;
    element.textContent = message;
    this.updateValidationList();
  }

  updateRunEnabled() {
    $("#runInference").disabled = !(this.state.sequence && this.state.model);
    this.updateValidationList();
  }

  updateValidationList() {
    const resultReady = Boolean(this.state.result);
    const sequenceReady = Boolean(this.state.sequence);
    const items = [
      [
        sequenceReady || resultReady,
        sequenceReady ? "Sequence loaded and valid" : "Completed run input recorded",
        "Sequence not validated",
      ],
      [Boolean(this.state.model), "Model checkpoint loaded", "Model checkpoint not loaded"],
      [resultReady, "Inference result available", "No completed inference"],
    ];
    $("#validationList").innerHTML = items
      .map(([ready, yes, no]) => `<li class="${ready ? "success" : "pending"}">${ready ? yes : no}</li>`)
      .join("");
  }

  annotationMode() {
    return document.querySelector("[name='annotationMode']:checked").value;
  }

  renderAnnotationMode() {
    const rnaMode = ["expression", "differential", "consensus_de"].includes(
      this.annotationMode(),
    );
    $("#rnaFilterControls").classList.toggle("hidden", !rnaMode);
    $("#tfFilterControl").classList.toggle(
      "hidden",
      this.annotationMode() !== "tf",
    );
  }

  renderAnnotationThresholds() {
    $("#logfcValue").textContent = Number($("#logfcThreshold").value).toFixed(1);
    $("#expressionValue").textContent = Number(
      $("#expressionThreshold").value,
    ).toLocaleString(undefined, { maximumFractionDigits: 1 });
    $("#padjValue").textContent = Number($("#padjThreshold").value).toFixed(2);
    $("#effectValue").textContent = `${Number($("#effectThreshold").value).toFixed(0)}%`;
  }

  configureAnnotationControls(summary) {
    const ranges = summary.rna_ranges || {};
    const logfcMaximum = Math.max(2, Math.ceil(ranges.abs_logfc_max || 6));
    $("#logfcThreshold").max = String(logfcMaximum);
    $("#logfcThreshold").value = String(Math.min(2, logfcMaximum));
    const expressionMaximum = Math.max(10, Math.ceil(ranges.expression_max || 1000));
    $("#expressionThreshold").max = String(expressionMaximum);
    $("#expressionThreshold").step = "1";
    $("#expressionThreshold").value = String(Math.min(10, expressionMaximum));
    $("#tfSelector").innerHTML = summary.available_tfs
      .map((tf) => `<option value="${escapeHTML(tf)}">${escapeHTML(tf)}</option>`)
      .join("");
    if (!summary.rna_columns && !summary.gene_list_count) {
      document.querySelector("[name='annotationMode'][value='tf']").checked = true;
    }
    this.renderAnnotationMode();
    this.renderAnnotationThresholds();
  }

  async loadAnnotations() {
    if (!this.state.result || !this.state.selectedJobId) return;
    $("#loadAnnotations").disabled = true;
    this.setValidation(
      $("#annotationStatus"),
      "loading",
      "Loading and validating motif/RNA tables…",
    );
    try {
      const payload = await api.post("/api/annotations/load", {
        job_id: this.state.selectedJobId,
        motif_path: $("#motifPath").value.trim(),
        rna_path: $("#rnaPath").value.trim(),
        rna_path_2: $("#rnaPath2").value.trim(),
        gene_list_path: $("#geneListPath").value.trim(),
        merge_overlaps: true,
      });
      this.state.annotationSummary = payload.annotations;
      this.configureAnnotationControls(payload.annotations);
      $("#annotationFilters").classList.remove("hidden");
      this.setValidation(
        $("#annotationStatus"),
        "success",
        `${payload.annotations.merged_motif_count.toLocaleString()} merged hits · `
          + `${payload.annotations.available_tfs.length.toLocaleString()} TFs`,
      );
      this.renderAnnotationDetails(payload.annotations);
      await this.filterAnnotations();
    } catch (error) {
      this.state.annotationSummary = null;
      this.motifTrack.clear();
      $("#annotationFilters").classList.add("hidden");
      this.setValidation($("#annotationStatus"), "error", error.message);
    } finally {
      $("#loadAnnotations").disabled = !this.state.result;
    }
  }

  annotationFilterPayload() {
    return {
      job_id: this.state.selectedJobId,
      mode: this.annotationMode(),
      min_abs_logfc: Number($("#logfcThreshold").value),
      min_expression: Number($("#expressionThreshold").value),
      max_padj: Number($("#padjThreshold").value),
      selected_tf: $("#tfSelector").value,
      min_effect_pct: Number($("#effectThreshold").value),
    };
  }

  scheduleAnnotationFilter() {
    if (!this.state.annotationSummary) return;
    window.clearTimeout(this.state.annotationFilterTimer);
    this.state.annotationFilterTimer = window.setTimeout(
      () => this.filterAnnotations(),
      120,
    );
  }

  async filterAnnotations() {
    if (!this.state.annotationSummary || !this.state.result) return;
    try {
      const payload = await api.post(
        "/api/annotations/filter",
        this.annotationFilterPayload(),
      );
      const filtered = payload.annotations;
      const groupStyle = this.selectedPreset()?.annotations?.group_style || "tf";
      this.motifTrack.setData(filtered.motifs, filtered.groups, groupStyle);
      $("#motifEmpty").classList.toggle("hidden", filtered.motif_count > 0);
      const modeLabels = {
        expression: `Expression: baseMean > ${filtered.filters.min_expression.toLocaleString()}`,
        differential: `Single DE: baseMean > ${filtered.filters.min_expression.toLocaleString()}, `
          + `|log2FC| > ${filtered.filters.min_abs_logfc}, padj < ${filtered.filters.max_padj}`,
        consensus_de: `Consensus DE: both tables pass baseMean/log2FC, same sign, `
          + `padj < ${filtered.filters.max_padj} in at least one`,
        gene_list: "Manuscript relaxed TF list",
        tf: `Direct TF: ${filtered.filters.selected_tf}`,
      };
      const modeText = modeLabels[filtered.mode] || filtered.mode;
      $("#motifSubtitle").textContent =
        `${filtered.motif_count.toLocaleString()} motifs · `
        + `${filtered.tf_count.toLocaleString()} TFs · `
        + `${filtered.group_count.toLocaleString()} plotted rows · ${modeText} · `
        + `deletion effect ≥ ${filtered.filters.min_effect_pct}%`;
      this.renderAnnotationDetails(this.state.annotationSummary, filtered);
    } catch (error) {
      this.setValidation($("#annotationStatus"), "error", error.message);
    }
  }

  renderAnnotationDetails(summary, filtered = null) {
    const columns = summary.rna_columns
      ? `${summary.rna_columns.gene} / `
        + `${summary.rna_columns.logfc || "—"} / `
        + `${summary.rna_columns.expression || "—"}`
      : "No RNA table";
    const filter = filtered
      ? `${filtered.mode.replaceAll("_", " ")} · ≥${filtered.filters.min_effect_pct}% effect`
      : "Loaded; awaiting filter";
    $("#annotationDetails").innerHTML = this.detailRows([
      [
        "Motifs",
        filtered
          ? `${filtered.motif_count} shown / ${summary.merged_motif_count} merged`
          : `${summary.merged_motif_count} merged`,
      ],
      [
        "Gene evidence",
        summary.gene_list_count
          ? `${summary.gene_list_count} genes + ${columns}`
          : columns,
      ],
      ["Filter", filter],
    ]);
  }

  resetAnnotations() {
    this.state.annotationSummary = null;
    this.motifTrack.clear();
    $("#motifEmpty").classList.remove("hidden");
    $("#motifSubtitle").textContent =
      "Load a motif table and optionally an RNA table after inference.";
    $("#annotationFilters").classList.add("hidden");
    $("#loadAnnotations").disabled = !this.state.result;
    this.setValidation(
      $("#annotationStatus"),
      "neutral",
      this.state.result
        ? "Choose motif/RNA files and load annotations"
        : "Complete a genomic-region run first",
    );
    $("#annotationDetails").innerHTML = this.detailRows([
      ["Motifs", "—"],
      ["RNA columns", "—"],
      ["Filter", "—"],
    ]);
  }

  deletionPayload() {
    return {
      deletion_size: Number($("#deletionSize").value),
      stride: Number($("#stride").value),
      batch_size: Number($("#batchSize").value),
      smoothing_window: Number($("#smoothing").value),
      perturbation: document.querySelector("[name='perturbation']:checked").value,
      output_head: $("#outputHead").value,
      scan_scope: $("#scanScope").value,
    };
  }

  async submitRun() {
    $("#runInference").disabled = true;
    this.state.result = null;
    this.resetAnnotations();
    this.state.startedAt = Date.now();
    let submittedJob = null;
    try {
      const payload = await api.post("/api/runs", {
        ...this.sequencePayload(),
        preset_id: $("#analysisPreset").value,
        deletion: this.deletionPayload(),
      });
      this.state.selectedJobId = payload.job.id;
      submittedJob = payload.job;
      await this.refreshJobs();
    } catch (error) {
      this.showRunError(error.message);
    } finally {
      this.updateRunEnabled();
    }
    return submittedJob;
  }

  async refreshJobs() {
    try {
      const payload = await api.get("/api/jobs");
      this.state.jobs = payload.jobs;
      this.renderJobs();
      const selected = this.state.jobs.find((job) => job.id === this.state.selectedJobId)
        || this.state.jobs[0];
      if (selected) {
        this.state.selectedJobId = selected.id;
        this.renderRunStatus(selected);
        if (selected.status === "complete" && (!this.state.result || this.state.result.jobId !== selected.id)) {
          await this.loadResult(selected.id);
        }
      }
    } catch (error) {
      this.showRunError(error.message);
    }
  }

  renderJobs() {
    if (!this.state.jobs.length) {
      $("#jobRows").innerHTML = '<tr><td colspan="6" class="empty-cell">No runs yet.</td></tr>';
      return;
    }
    $("#jobRows").innerHTML = this.state.jobs.map((job) => {
      const progress = job.total ? `${job.completed.toLocaleString()} / ${job.total.toLocaleString()}` : "—";
      const action = job.status === "running"
        ? `<button class="job-action" data-cancel="${job.id}">Cancel</button>`
        : `<button class="job-action" data-select="${job.id}">View</button>`;
      return `<tr>
        <td>${job.id}</td>
        <td class="status-text ${job.status}">${job.status}</td>
        <td>${job.stage}</td>
        <td>${progress}</td>
        <td>${job.started_at || "—"}</td>
        <td>${action}</td>
      </tr>`;
    }).join("");
    $$("[data-select]").forEach((button) => button.addEventListener("click", async () => {
      this.state.selectedJobId = button.dataset.select;
      const job = this.state.jobs.find((item) => item.id === this.state.selectedJobId);
      this.renderRunStatus(job);
      if (job.status === "complete") await this.loadResult(job.id);
    }));
    $$("[data-cancel]").forEach((button) => button.addEventListener("click", async () => {
      await api.post(`/api/jobs/${button.dataset.cancel}/cancel`, {});
      await this.refreshJobs();
    }));
  }

  renderRunStatus(job) {
    $("#runState").className = `run-state ${job.status}`;
    $("#runState").textContent = job.status[0].toUpperCase() + job.status.slice(1);
    $("#runStage").textContent = job.stage;
    const percent = job.total ? Math.round((job.completed / job.total) * 100) : 0;
    $("#progressBar").style.width = `${percent}%`;
    $("#runProgress").textContent = job.total ? `${job.completed.toLocaleString()} / ${job.total.toLocaleString()} (${percent}%)` : "—";
    if (job.started_at) {
      const elapsed = job.finished_at
        ? (new Date(job.finished_at) - new Date(job.started_at)) / 1000
        : (Date.now() - new Date(job.started_at)) / 1000;
      $("#runElapsed").textContent = this.formatDuration(elapsed);
    }
    if (job.error) this.showRunError(job.error);
    else $("#runError").classList.add("hidden");
  }

  async loadResult(jobId) {
    const previousJobId = this.state.result?.jobId;
    const result = await api.get(`/api/jobs/${jobId}/result`);
    result.jobId = jobId;
    this.state.result = result;
    if (previousJobId !== jobId) this.resetAnnotations();
    $("#deletionSize").value = result.settings.deletion_size;
    $("#stride").value = result.settings.stride;
    $("#batchSize").value = result.settings.batch_size;
    $("#smoothing").value = result.settings.smoothing_window;
    $("#scanScope").value = result.settings.scan_scope || "context";
    const perturbation = document.querySelector(
      `[name="perturbation"][value="${result.settings.perturbation}"]`,
    );
    if (perturbation) perturbation.checked = true;
    if (Array.from($("#outputHead").options).some((option) => option.value === result.settings.output_head)) {
      $("#outputHead").value = result.settings.output_head;
    }
    $("#modelDetails").innerHTML = this.detailRows([
      ["Deletion model", result.model.checkpoint.split("/").pop()],
      [
        "Profile model",
        result.profile_model
          ? result.profile_model.checkpoint.split("/").pop()
          : "same / not configured",
      ],
      ["Device", result.model.device],
      ["Output heads", result.model.output_heads.join(", ")],
    ]);
    const regionLabels = Object.fromEntries(
      Object.entries(result.regions || {}).map(
        ([name, value]) => [name, value?.label || "—"],
      ),
    );
    this.renderInputDetails({
      ...result.sequence,
      regions: regionLabels,
    });
    const head = result.settings.output_head;
    const baseline = result.baseline[head];
    const sequenceStart = result.sequence.start || 0;
    const display = result.regions?.display || result.sequence;
    let profilePositions;
    let profileSeries;
    if (result.profile?.predicted) {
      profilePositions = result.profile.positions;
      profileSeries = [
        {
          label: "Current checkpoint",
          values: result.profile.predicted,
          color: "#007181",
        },
      ];
      if (
        Array.isArray(result.profile.observed)
        && result.profile.observed.length === profilePositions.length
      ) {
        profileSeries.push({
          label: "Observed ATAC",
          values: result.profile.observed,
          color: "#7253a3",
          alpha: 0.82,
        });
      }
    } else {
      profilePositions = this.profilePositions(
        sequenceStart,
        result.sequence.length,
        baseline.length,
      );
      profileSeries = [{
        label: head,
        values: baseline,
        color: "#007181",
      }];
    }
    const displayedProfile = this.sliceSeries(
      profilePositions,
      profileSeries,
      display.start,
      display.end,
    );
    const axisStart = Number(display.start);
    const axisEnd = Number(display.end) - 1;
    this.baselineChart.setDomain(axisStart, axisEnd);
    this.deletionChart.setDomain(axisStart, axisEnd);
    this.baselineChart.setSeries(
      displayedProfile.positions,
      displayedProfile.series,
    );
    const transformed = result.plot_effects
      ? {
          values: result.plot_effects,
          metadata: result.plot_effect,
        }
      : manuscriptPlotEffects(result.nucleotide_effects, baseline);
    this.deletionChart.setYRange(transformed.metadata.applied ? [0, 100] : null);
    const displayedDeletion = this.sliceSeries(
      result.nucleotide_positions,
      [{ label: "Deletion effect", values: transformed.values, color: "#c33a35" }],
      display.start,
      display.end,
    );
    this.deletionChart.setSeries(
      displayedDeletion.positions,
      displayedDeletion.series,
    );
    this.motifTrack.setDomain(
      axisStart,
      axisEnd,
    );
    $("#baselineEmpty").classList.add("hidden");
    $("#deletionEmpty").classList.add("hidden");
    $("#baselineSubtitle").textContent = result.profile_model
      ? `Model prediction and observed ATAC · ${display.label}`
      : `${head} · ${display.label || result.sequence.label}`;
    $("#deletionSubtitle").textContent = transformed.metadata.applied
      ? `${result.settings.deletion_size}-bp deletion · `
        + `effect as % of maximum predicted accessibility · ${display.label}`
      : `${transformed.metadata.label} · normalization unavailable`;
    const genomicResult = Boolean(
      result.sequence.chromosome
      && Number.isFinite(result.sequence.start)
      && Number.isFinite(result.sequence.end),
    );
    $("#loadAnnotations").disabled = !genomicResult;
    if (!genomicResult) {
      this.setValidation(
        $("#annotationStatus"),
        "neutral",
        "Motif annotations require a genomic-region run",
      );
    }
    $("#downloadJson").disabled = false;
    $("#downloadCsv").disabled = false;
    this.updateValidationList();
  }

  profilePositions(start, sequenceLength, profileLength) {
    if (profileLength <= 1) return [start];
    const scale = (sequenceLength - 1) / (profileLength - 1);
    return Array.from({ length: profileLength }, (_, index) => Math.round(start + index * scale));
  }

  sliceSeries(positions, series, start, end) {
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      return { positions, series };
    }
    const indices = positions
      .map((position, index) => ({ position: Number(position), index }))
      .filter((item) => item.position >= start && item.position < end);
    return {
      positions: indices.map((item) => item.position),
      series: series.map((item) => ({
        ...item,
        values: indices.map(({ index }) => item.values[index]),
      })),
    };
  }

  download(name) {
    if (!this.state.selectedJobId) return;
    window.location.assign(`/api/jobs/${this.state.selectedJobId}/${name}`);
  }

  showRunError(message) {
    $("#runError").textContent = message;
    $("#runError").classList.remove("hidden");
  }

  formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.max(0, Math.round(seconds % 60));
    return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
  }
}

const app = new ModelStudioApp();
app.initialize().catch((error) => {
  console.error(error);
  $("#modelStatus").className = "validation error";
  $("#modelStatus").textContent = error.message;
});
