"use client";

// Renders the markdown the filing and earnings summaries are written in:
// pipe tables, bullet lists, **bold**, <br> inside table cells, and paragraphs.
// Anything else is shown as plain text. Text goes through React, so nothing
// in a summary is ever rendered as HTML.

type Block =
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "list"; items: string[] }
  | { kind: "para"; text: string };

const BULLET = /^(?:[-*•])\s+/;
const DIVIDER = /^\|?\s*:?-{2,}/;

function splitRow(line: string): string[] {
  return line
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());
}

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line) {
      i++;
      continue;
    }
    if (line.startsWith("|") && i + 1 < lines.length && DIVIDER.test(lines[i + 1].trim())) {
      const header = splitRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(splitRow(lines[i].trim()));
        i++;
      }
      blocks.push({ kind: "table", header, rows });
      continue;
    }
    if (BULLET.test(line)) {
      const items: string[] = [];
      while (i < lines.length && BULLET.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(BULLET, ""));
        i++;
      }
      blocks.push({ kind: "list", items });
      continue;
    }
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("|") &&
      !BULLET.test(lines[i].trim())
    ) {
      para.push(lines[i].trim());
      i++;
    }
    blocks.push({ kind: "para", text: para.join(" ") });
  }
  return blocks;
}

// **bold** and <br> inline; everything else is literal text.
function renderInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*|<br\s*\/?>)/gi).map((part, idx) => {
    if (!part) return null;
    if (/^<br/i.test(part)) return <br key={idx} />;
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={idx} className="font-semibold text-slate-900">{part.slice(2, -2)}</strong>;
    }
    return <span key={idx}>{part}</span>;
  });
}

export default function FilingSummaryText({ text }: { text: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className="flex flex-col gap-3 text-sm leading-relaxed text-slate-800">
      {blocks.map((b, idx) => {
        if (b.kind === "table") {
          return (
            <div key={idx} className="overflow-x-auto rounded-md border border-slate-200">
              <table className="w-full min-w-[32rem] border-collapse text-left text-xs">
                <thead className="bg-slate-50">
                  <tr>
                    {b.header.map((h, j) => (
                      <th key={j} className="border-b border-slate-200 px-3 py-2 font-semibold text-slate-700">
                        {renderInline(h)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {b.rows.map((r, j) => (
                    <tr key={j} className="align-top odd:bg-white even:bg-slate-50/50">
                      {r.map((c, k) => (
                        <td key={k} className="border-b border-slate-100 px-3 py-2 text-slate-800">
                          {renderInline(c)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        if (b.kind === "list") {
          return (
            <ul key={idx} className="flex list-disc flex-col gap-1 pl-5">
              {b.items.map((item, j) => (
                <li key={j}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }
        return <p key={idx}>{renderInline(b.text)}</p>;
      })}
    </div>
  );
}
