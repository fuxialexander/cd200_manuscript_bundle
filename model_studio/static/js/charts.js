const formatNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });

export class LineChart {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.options = {
      color: options.color || "#007181",
      zeroLine: options.zeroLine ?? false,
      yLabel: options.yLabel || "",
      yRange: options.yRange || null,
    };
    this.data = null;
    this.range = null;
    this.domain = null;
    this.tooltip = options.tooltip || null;
    this.resizeObserver = new ResizeObserver(() => this.render());
    this.resizeObserver.observe(canvas.parentElement);
    canvas.addEventListener("mousemove", (event) => this.showTooltip(event));
    canvas.addEventListener("mouseleave", () => this.hideTooltip());
  }

  setData(x, y) {
    this.setSeries(x, [{ label: "Value", values: y, color: this.options.color }]);
  }

  setSeries(x, series) {
    this.data = {
      x,
      series: series.filter((item) => Array.isArray(item.values)),
    };
    this.range = [0, Math.max(0, x.length - 1)];
    this.render();
  }

  setYRange(range) {
    this.options.yRange = range;
    this.render();
  }

  setDomain(start, end) {
    this.domain = [Number(start), Number(end)];
    this.render();
  }

  resetZoom() {
    if (this.data) this.range = [0, Math.max(0, this.data.x.length - 1)];
    this.render();
  }

  clear() {
    this.data = null;
    this.render();
  }

  dimensions() {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: rect.width, height: rect.height };
  }

  visibleData() {
    if (!this.data) return null;
    const [start, end] = this.range;
    return {
      x: this.data.x.slice(start, end + 1),
      series: this.data.series.map((item) => ({
        ...item,
        values: item.values.slice(start, end + 1),
      })),
      offset: start,
    };
  }

  render() {
    const { width, height } = this.dimensions();
    const ctx = this.context;
    ctx.clearRect(0, 0, width, height);
    const visible = this.visibleData();
    if (!visible || visible.x.length === 0 || visible.series.length === 0) return;

    const margin = { left: 92, right: 46, top: 10, bottom: 29 };
    const plotWidth = Math.max(1, width - margin.left - margin.right);
    const plotHeight = Math.max(1, height - margin.top - margin.bottom);
    const allValues = visible.series.flatMap((item) =>
      item.values.filter(Number.isFinite));
    if (!allValues.length) return;
    let minY = Math.min(...allValues);
    let maxY = Math.max(...allValues);
    if (this.options.yRange) {
      [minY, maxY] = this.options.yRange;
    } else if (this.options.zeroLine) {
      minY = Math.min(minY, 0);
      maxY = Math.max(maxY, 0);
    }
    if (minY === maxY) {
      minY -= 0.5;
      maxY += 0.5;
    }
    if (!this.options.yRange) {
      const pad = (maxY - minY) * 0.08;
      minY -= pad;
      maxY += pad;
    }

    const sampledStart = Number(visible.x[0]);
    const sampledEnd = Number(visible.x[visible.x.length - 1]);
    const domainStart = Number.isFinite(this.domain?.[0])
      ? this.domain[0]
      : sampledStart;
    const domainEnd = Number.isFinite(this.domain?.[1])
      ? this.domain[1]
      : sampledEnd;
    const domainWidth = Math.max(1, domainEnd - domainStart);
    const xAt = (index) =>
      margin.left
      + ((Number(visible.x[index]) - domainStart) / domainWidth) * plotWidth;
    const yAt = (value) =>
      margin.top + (1 - (value - minY) / (maxY - minY)) * plotHeight;

    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#e1e7e9";
    ctx.fillStyle = "#617178";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick += 1) {
      const fraction = tick / 4;
      const y = margin.top + fraction * plotHeight;
      const value = maxY - fraction * (maxY - minY);
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(width - margin.right, y);
      ctx.stroke();
      ctx.fillText(formatNumber.format(value), margin.left - 7, y);
    }

    if (this.options.zeroLine && minY < 0 && maxY > 0) {
      ctx.save();
      ctx.strokeStyle = "#87959a";
      ctx.setLineDash([4, 4]);
      const zeroY = yAt(0);
      ctx.beginPath();
      ctx.moveTo(margin.left, zeroY);
      ctx.lineTo(width - margin.right, zeroY);
      ctx.stroke();
      ctx.restore();
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTickIntervals = width < 520 ? 2 : 4;
    for (let tick = 0; tick <= xTickIntervals; tick += 1) {
      const fraction = tick / xTickIntervals;
      const value = domainStart + fraction * domainWidth;
      const x = margin.left + fraction * plotWidth;
      ctx.fillText(
        formatNumber.format(Math.round(value)),
        x,
        height - margin.bottom + 8,
      );
    }

    visible.series.forEach((series) => {
      ctx.beginPath();
      series.values.forEach((value, index) => {
        if (!Number.isFinite(value)) return;
        const x = xAt(index);
        const y = yAt(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = series.color || this.options.color;
      ctx.lineWidth = series.lineWidth || 1.6;
      ctx.globalAlpha = series.alpha ?? 1;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    this.canvas.dataset.domainStart = String(domainStart);
    this.canvas.dataset.domainEnd = String(domainEnd);
    this.canvas.dataset.plotLeft = String(margin.left);
    this.canvas.dataset.plotRight = String(margin.right);
    this.scale = {
      margin,
      plotWidth,
      plotHeight,
      minY,
      maxY,
      visible,
      xAt,
      yAt,
      domainStart,
      domainWidth,
    };
  }

  showTooltip(event) {
    if (!this.tooltip || !this.scale) return;
    const rect = this.canvas.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const {
      margin,
      plotWidth,
      visible,
      xAt,
      yAt,
      domainStart,
      domainWidth,
    } = this.scale;
    if (localX < margin.left || localX > margin.left + plotWidth) {
      this.hideTooltip();
      return;
    }
    const index = Math.max(
      0,
      Math.min(
        visible.x.length - 1,
        Math.round(
          (
            (
              domainStart
              + ((localX - margin.left) / plotWidth) * domainWidth
              - Number(visible.x[0])
            )
            / Math.max(
              1,
              Number(visible.x[visible.x.length - 1]) - Number(visible.x[0]),
            )
          )
          * (visible.x.length - 1),
        ),
      ),
    );
    const x = visible.x[index];
    const rows = visible.series.map((series) => {
      const value = series.values[index];
      return `${series.label}: ${formatNumber.format(value)}`;
    });
    const y = visible.series[0].values[index];
    this.tooltip.innerHTML = `<strong>Position ${formatNumber.format(x)}</strong><br>${rows.join("<br>")}`;
    this.tooltip.style.left = `${Math.min(rect.width - 165, xAt(index) + 10)}px`;
    this.tooltip.style.top = `${Math.max(4, yAt(y) - 48)}px`;
    this.tooltip.classList.remove("hidden");
  }

  hideTooltip() {
    if (this.tooltip) this.tooltip.classList.add("hidden");
  }
}
