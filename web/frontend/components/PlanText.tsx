// Renders the lightweight markdown produced by services/portfolio_strategy.py
// (headings as **bold** lines, "- " bullet lists, `code` prices) without pulling
// in a full markdown dependency — the format is small and fixed.

function renderInline(text: string, keyPrefix: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={`${keyPrefix}-${i}`} className="rounded bg-slate-200/70 px-1 py-0.5 font-mono text-xs text-slate-800">
          {part.slice(1, -1)}
        </code>
      );
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>;
  });
}

export default function PlanText({ text }: { text: string }) {
  const blocks = text.trim().split(/\n\n+/);

  return (
    <div className="space-y-2">
      {blocks.map((block, bi) => {
        const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
        if (lines.length === 0) return null;

        const isTitleOnly = lines.length === 1 && /^\*\*.+\*\*$/.test(lines[0]);
        if (isTitleOnly) {
          return (
            <h4 key={bi} className="text-sm font-semibold text-slate-900">
              {lines[0].slice(2, -2)}
            </h4>
          );
        }

        const isList = lines.every((l) => l.startsWith("- "));
        if (isList) {
          return (
            <ul key={bi} className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-slate-700">
              {lines.map((l, li) => (
                <li key={li}>{renderInline(l.replace(/^- /, ""), `${bi}-${li}`)}</li>
              ))}
            </ul>
          );
        }

        return (
          <p key={bi} className="text-sm leading-relaxed text-slate-700">
            {renderInline(lines.join(" "), `${bi}`)}
          </p>
        );
      })}
    </div>
  );
}
