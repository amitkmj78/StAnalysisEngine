// BEG-2: Learning Paths. Static content (ships with the frontend bundle,
// not stored server-side) — only completion/quiz-score tracking lives in
// the backend (lesson_progress table, web/backend/routers/learning.py).
// General education, not personalized financial advice — same framing this
// app already uses elsewhere (e.g. trade_impact.py's "not advice" note).

export interface QuizQuestion {
  question: string;
  choices: string[];
  correctIndex: number;
}

export interface Lesson {
  id: string;
  title: string;
  summary: string;
  body: string[];
  quiz: QuizQuestion[];
  liveExample: { href: string; label: string };
  // Some topics already have a deeper, app-specific explainer under
  // /guides (how THIS app computes/flags the thing, not general
  // education) -- link to it when one exists, rather than duplicating it.
  guideHref?: string;
}

export const LESSONS: Lesson[] = [
  {
    id: "what-is-a-stock",
    title: "What a stock is",
    summary: "What you actually own when you buy a share.",
    body: [
      "A share of stock is a small piece of ownership in a company. If the company does well — grows its sales, earns more profit — the stock often becomes more valuable over time, because owning a piece of a more valuable company is worth more. If it does poorly, the opposite can happen.",
      "A stock's price moves for two broad reasons: what's actually happening at the company (earnings, growth, news), and how the overall market is feeling that day or year (interest rates, the economy, investor mood). Short-term price moves are often more about the second thing than the first.",
      "Buying a stock does not make you a lender to the company and does not guarantee you anything — there's no promised return, and a company can lose money or go to zero. That's the trade-off for the upside: real ownership, real risk.",
    ],
    quiz: [
      {
        question: "When you buy one share of a company, you become:",
        choices: [
          "A lender who the company must pay back",
          "A small part-owner of the company",
          "An employee of the company",
          "Guaranteed a fixed yearly payment",
        ],
        correctIndex: 1,
      },
      {
        question: "A stock's price can move even when nothing has changed at the company itself. Why?",
        choices: [
          "It can't — price only moves on company news",
          "Overall market mood (rates, economy, sentiment) affects prices too",
          "Prices only update once a year",
          "Only employees can move the price",
        ],
        correctIndex: 1,
      },
    ],
    liveExample: { href: "/predict", label: "See a real forecast on Predict" },
  },
  {
    id: "diversification",
    title: "Diversification",
    summary: "Why \"don't put all your eggs in one basket\" is more than a saying.",
    body: [
      "Diversification means spreading money across many different investments instead of concentrating it in one or two. If one stock drops sharply, a diversified portfolio feels it as a small dent; a concentrated one feels it as a gut punch.",
      "This app's Health Check flags when one position is an outsized share of your total portfolio, or when several holdings are all exposed to the same sector — both are forms of concentration, even if they don't feel that way while things are going well.",
      "Diversification doesn't eliminate risk and doesn't guarantee a better return — on a great year for one stock, concentration can outperform. What it does is reduce the odds that a single bad surprise does lasting damage.",
    ],
    quiz: [
      {
        question: "The main benefit of diversification is:",
        choices: [
          "It guarantees higher returns",
          "It reduces how much a single bad surprise can hurt the whole portfolio",
          "It eliminates all investment risk",
          "It only matters for very large portfolios",
        ],
        correctIndex: 1,
      },
      {
        question: "Two stocks in different sectors, but which move for the same underlying reason, are:",
        choices: [
          "Automatically well diversified because they're in different sectors",
          "Still concentrated risk, if that shared reason goes wrong",
          "Irrelevant to diversification",
          "Only a concern for bonds",
        ],
        correctIndex: 1,
      },
    ],
    liveExample: { href: "/portfolio/health", label: "Check your own concentration on Health Check" },
    guideHref: "/guides/diversification",
  },
  {
    id: "risk-and-drawdown",
    title: "Risk and drawdown",
    summary: "How to think about how much a portfolio can fall — before it happens.",
    body: [
      "A drawdown is how far a portfolio has fallen from its most recent peak, usually shown as a percentage. Every investor experiences drawdowns — the real question is how large a drawdown you can sit through without panic-selling at the worst possible time.",
      "Risk and expected return are linked: investments that can swing more sharply (higher volatility) have historically offered the chance of higher returns too, as compensation for that bumpier ride. There's no free lunch — a path to higher expected return without more risk is rare and worth being skeptical of.",
      "A useful habit before buying anything: ask \"if this fell 20-30%, would I hold on, or would I need to sell?\" If the honest answer is you'd need to sell, the position may be sized too large for your own comfort, regardless of how good the opportunity looks.",
    ],
    quiz: [
      {
        question: "A \"drawdown\" measures:",
        choices: [
          "How much a stock pays in dividends",
          "How far a portfolio has fallen from its most recent peak",
          "How many shares you've sold",
          "The company's quarterly revenue",
        ],
        correctIndex: 1,
      },
      {
        question: "Generally speaking, investments capable of larger price swings (higher volatility) have:",
        choices: [
          "No relationship to expected return at all",
          "Always lower expected returns",
          "Historically offered the chance of higher returns, as compensation for the risk",
          "Guaranteed higher returns",
        ],
        correctIndex: 2,
      },
      {
        question: "Before buying, a useful question to ask yourself is:",
        choices: [
          "Will this definitely go up?",
          "Could I hold through a 20-30% drop in this position without panic-selling?",
          "What did it do yesterday?",
          "Is the ticker symbol easy to remember?",
        ],
        correctIndex: 1,
      },
    ],
    liveExample: { href: "/portfolio/stress-test", label: "Try a historical stress test on your portfolio" },
    guideHref: "/guides/risk",
  },
  {
    id: "reading-a-signal",
    title: "Reading a signal",
    summary: "What \"BUY / HOLD / SELL\" from this app's model actually means.",
    body: [
      "This app's model looks at a stock's recent price and technical history and forecasts an expected return over a set horizon (e.g. 10 days). BUY means the forecast implies a meaningfully positive expected return; SELL means a meaningfully negative one; HOLD means the forecast is roughly in between — not a strong edge either way.",
      "A signal is a model output, not a guarantee and not personalized advice. It can be wrong, and it says nothing about your own goals, time horizon, or how much of this stock you already hold.",
      "Treat a signal as one input among several — your own research, your diversification, and your own tolerance for risk all matter too. This app's trade-preview guardrails (shown before you place a paper trade) exist specifically to surface when a trade would go against the model's current signal, so you can at least make that choice knowingly.",
    ],
    quiz: [
      {
        question: "A BUY signal from this app's model means:",
        choices: [
          "The stock is guaranteed to go up",
          "The forecast implies a meaningfully positive expected return over its horizon",
          "An analyst has personally recommended it",
          "It is risk-free",
        ],
        correctIndex: 1,
      },
      {
        question: "A signal should be treated as:",
        choices: [
          "Personalized financial advice",
          "One input among several, not a guarantee",
          "A promise about future performance",
          "Irrelevant and safe to ignore entirely",
        ],
        correctIndex: 1,
      },
    ],
    liveExample: { href: "/predict", label: "Look up a signal for a real ticker" },
    guideHref: "/guides/signals",
  },
  {
    id: "stop-losses",
    title: "Stop-losses",
    summary: "A plan for when to walk away, decided before emotions are involved.",
    body: [
      "A stop-loss is a price level you decide on in advance: if the position falls to that price, you sell, rather than deciding in the moment. The point is to make the decision when you're calm, not when the position is down and every instinct says \"wait, it might come back.\"",
      "This app's Short-/Long-Term Plan on the Portfolio page already computes a suggested protective stop price per position, based on your risk settings and that ticker's own recent volatility — a concrete number to react to, not just a vague feeling.",
      "A stop-loss isn't free of trade-offs: it can trigger a sale right before a bounce, locking in a loss that would have reversed. It's a risk-management tool, not a way to guarantee better outcomes — it trades away some upside for a cap on how bad a single loss can get.",
    ],
    quiz: [
      {
        question: "The main purpose of a stop-loss is to:",
        choices: [
          "Guarantee a profit",
          "Decide your exit point in advance, before emotions take over",
          "Avoid ever having a loss",
          "Predict the exact bottom of a decline",
        ],
        correctIndex: 1,
      },
      {
        question: "A real trade-off of using a stop-loss is:",
        choices: [
          "There is no trade-off, it only helps",
          "It can sell right before a price bounces back",
          "It always increases your return",
          "It's only usable on Mondays",
        ],
        correctIndex: 1,
      },
    ],
    liveExample: { href: "/portfolio", label: "See your own positions' suggested stop prices" },
  },
  {
    id: "costs-and-taxes",
    title: "Costs and taxes",
    summary: "The quiet costs that eat into real returns.",
    body: [
      "Every trade and every holding has some cost: a fund's expense ratio, a brokerage's fees, and (in a taxable account) taxes owed on realized gains and on dividends received. None of these show up as a dramatic loss in the moment, which is exactly why they're easy to ignore and easy to underestimate over years.",
      "Short-term capital gains (positions held about a year or less) are typically taxed at a higher rate than long-term gains in many places — meaning how long you hold, not just what you hold, affects what you actually keep. Rules vary and this isn't tax advice; check your own situation or a professional for specifics.",
      "This app's Health Check surfaces a portfolio's fee drag and tax-loss-harvesting opportunities (realizing a loss on purpose to offset a gain elsewhere) — small, boring-looking numbers that compound into a real difference over a long holding period.",
    ],
    quiz: [
      {
        question: "Why are ongoing costs (fees, taxes) easy to underestimate?",
        choices: [
          "They don't actually affect returns",
          "They show up as one dramatic loss",
          "They're quiet and gradual, but compound over years",
          "They only apply to retirement accounts",
        ],
        correctIndex: 2,
      },
      {
        question: "Holding period (short-term vs. long-term) matters mainly because:",
        choices: [
          "It has no effect on anything",
          "In many places, it affects the tax rate owed on a realized gain",
          "Longer holds are always required by law",
          "It changes the company's stock price directly",
        ],
        correctIndex: 1,
      },
    ],
    liveExample: { href: "/portfolio/health", label: "See fee drag and tax-loss opportunities on Health Check" },
  },
];

export function getLesson(id: string): Lesson | undefined {
  return LESSONS.find((l) => l.id === id);
}
