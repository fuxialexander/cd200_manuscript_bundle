const SVG_NS = "http://www.w3.org/2000/svg";

const createSVG = (name, attributes = {}) => {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
};

const formatValue = (value, digits = 2) =>
  Number.isFinite(value) ? Number(value).toFixed(digits) : "not available";

export class MotifTrack {
  constructor(container) {
    this.container = container;
    this.domain = null;
    this.motifs = [];
    this.groups = [];
    this.style = "tf";
    this.maxRows = 12;
    this.resizeObserver = new ResizeObserver(() => this.render());
    this.resizeObserver.observe(container);
  }

  setDomain(start, end) {
    this.domain = [Number(start), Number(end)];
    this.render();
  }

  setData(motifs, groups = [], style = "tf") {
    this.motifs = Array.isArray(motifs) ? motifs : [];
    this.groups = Array.isArray(groups) ? groups : [];
    this.style = style;
    this.render();
  }

  clear() {
    this.motifs = [];
    this.groups = [];
    this.render();
  }

  rankedTFs() {
    const scores = new Map();
    this.motifs.forEach((motif) => {
      scores.set(
        motif.tf,
        Math.max(scores.get(motif.tf) || 0, motif.overlap_effect_pct || 0),
      );
    });
    return Array.from(scores)
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, this.maxRows)
      .map(([tf]) => tf);
  }

  render() {
    this.container.replaceChildren();
    if (!this.domain || !this.motifs.length) return;
    const [domainStart, domainEnd] = this.domain;
    const figureRows = this.style === "figure" && this.groups.length
      ? this.groups.slice(0, this.maxRows)
      : null;
    const tfs = figureRows ? figureRows.map((group) => group.label) : this.rankedTFs();
    const width = Math.max(1, this.container.clientWidth || 360);
    const labelWidth = 92;
    const rightMargin = 46;
    const topMargin = 12;
    const rowHeight = 25;
    const bottomMargin = 24;
    const height = topMargin + tfs.length * rowHeight + bottomMargin;
    const plotWidth = width - labelWidth - rightMargin;
    const x = (position) =>
      labelWidth
      + ((Number(position) - domainStart) / Math.max(1, domainEnd - domainStart))
      * plotWidth;
    const svg = createSVG("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Filtered motif annotations",
      "data-domain-start": domainStart,
      "data-domain-end": domainEnd,
      "data-plot-left": labelWidth,
      "data-plot-right": rightMargin,
    });
    svg.classList.add("motif-svg");

    const xTickIntervals = width < 520 ? 2 : 4;
    for (let tick = 0; tick <= xTickIntervals; tick += 1) {
      const position = domainStart
        + ((domainEnd - domainStart) * tick) / xTickIntervals;
      const tickX = x(position);
      svg.appendChild(createSVG("line", {
        x1: tickX,
        x2: tickX,
        y1: topMargin,
        y2: height - bottomMargin,
        class: "motif-gridline",
      }));
      const label = createSVG("text", {
        x: tickX,
        y: height - 7,
        class: "motif-axis-label",
        "text-anchor": "middle",
      });
      label.textContent = Math.round(position).toLocaleString();
      svg.appendChild(label);
    }

    tfs.forEach((tf, rowIndex) => {
      const centerY = topMargin + rowIndex * rowHeight + rowHeight / 2;
      const label = createSVG("text", {
        x: labelWidth - 9,
        y: centerY + 4,
        class: "motif-tf-label",
        "text-anchor": "end",
      });
      label.textContent = tf;
      svg.appendChild(label);
      svg.appendChild(createSVG("line", {
        x1: labelWidth,
        x2: width - rightMargin,
        y1: centerY,
        y2: centerY,
        class: "motif-rowline",
      }));
      const rowMotifs = figureRows
        ? figureRows[rowIndex].motifs
        : this.motifs.filter((motif) => motif.tf === tf);
      rowMotifs
        .forEach((motif) => {
          const startX = Math.max(labelWidth, x(motif.start));
          const endX = Math.min(width - rightMargin, x(motif.end));
          const colorClass = Number.isFinite(motif.rna_logfc)
            ? (motif.rna_logfc >= 0 ? "rna-up" : "rna-down")
            : "rna-missing";
          const rect = createSVG("rect", {
            x: startX,
            y: centerY - 7,
            width: Math.max(2, endX - startX),
            height: 14,
            rx: 1,
            class: `motif-hit ${colorClass}`,
          });
          const title = createSVG("title");
          title.textContent = [
            figureRows
              ? `${figureRows[rowIndex].full_label} · ${motif.motif_name}`
              : `${motif.tf} · ${motif.motif_name}`,
            `${motif.chromosome}:${motif.start.toLocaleString()}-${motif.end.toLocaleString()}`,
            `Deletion effect: ${formatValue(motif.overlap_effect_pct)}%`,
            `RNA log2FC: ${formatValue(motif.rna_logfc)}`,
            `Expression: ${formatValue(motif.rna_expression)}`,
            `Merged hits: ${motif.hit_count}`,
          ].join("\n");
          rect.appendChild(title);
          svg.appendChild(rect);
        });
    });
    this.container.appendChild(svg);
  }
}
