import type { ColumnInfo } from "@/components/InfoModal";

// Plain-language explanations for the strategy builder, written for someone new to trading.
// They explain what each part means. None of them tells you to buy or sell anything.
export const STRATEGY_INFO: Record<string, ColumnInfo> = {
  template: {
    title: "Starting template",
    body: [
      "A ready-made set of rules to start from. Pick one, then change anything you like.",
      "Each template tries a different idea, so the results are not directly comparable.",
    ],
  },
  universe: {
    title: "Stocks to test",
    body: [
      "The companies the test buys and sells, up to 20. Each one gets the same amount of money.",
      "If you pick the stocks yourself, the result may depend on your choices rather than the rules. The random-sample re-run shows whether that is the case.",
    ],
  },
  presets: {
    title: "Quick stock lists",
    body: [
      "Random S&P 500: a random set of large US companies. The same random list can be drawn again later.",
      "Sector: the largest companies in one industry, such as Energy or Health Care.",
    ],
  },
  "plain words": {
    title: "What the rules say",
    body: ["The rules written as a sentence, so you can check what the test will do before you run it."],
  },
  entry: {
    title: "When to buy",
    body: [
      "A stock is bought when all of these are true at the end of a trading day. The purchase happens at the next day's opening price.",
      "“Crosses above” means it happened today and was not true yesterday. Use it so a buy happens once, not every day the condition stays true.",
    ],
  },
  "crosses above": {
    title: "Crosses above",
    body: ["True only on the day something moves up through a level. It fires once for each move, so it does not keep buying."],
  },
  "is above": {
    title: "Is above",
    body: ["True every day the value stays above the level. As a buy rule it can buy again right after a sale, so “crosses above” is usually better."],
  },
  exit: {
    title: "When to sell",
    body: [
      "If any one of these is true at the end of a day, the stock is sold at the next day's opening price.",
      "If a sell level is also a buy level, the stock is bought straight back. The warning under the rule points this out.",
    ],
  },
  "protective exit": {
    title: "Protective exit (loss limit)",
    body: [
      "A rule that sells if a stock falls too far, so one bad position cannot do too much damage.",
      "The test needs one of these, or a sell rule, unless you tick the box to run without one.",
    ],
  },
  "trailing stop": {
    title: "Trailing stop",
    body: [
      "Sells if the price drops by this percentage from its highest point since you bought.",
      "It follows the price up but never moves down, so it locks in some of the gain.",
    ],
  },
  "stop-loss": {
    title: "Stop-loss",
    body: ["Sells if the price drops by this percentage below the price you paid."],
  },
  "time stop": {
    title: "Time limit",
    body: ["Sells after this many trading days, whatever the price has done."],
  },
  "run without a protective exit": {
    title: "Running without a loss limit",
    body: ["Losing trades are kept until a sell rule fires. The result will warn you about this."],
  },
  cooldown: {
    title: "Wait before buying again",
    body: [
      "After selling a stock, the test waits this many trading days before it can buy that stock again.",
      "Without a wait, a stock can be sold and bought back the next day, which costs money for no real change.",
    ],
  },
  sizing: {
    title: "How much money per stock",
    body: ["Every stock gets the same share of the money. Other ways of sizing are not available yet."],
  },
  costs: {
    title: "Trading costs",
    body: [
      "Each buy and each sell is charged a small fee and a small price gap, about 0.15% in total. These are included in every result.",
    ],
  },
  fills: {
    title: "When trades happen",
    body: ["Rules are checked at the end of each day. Trades happen at the next day's opening price, which is a price you could really get."],
  },
  verdict: {
    title: "Verdict",
    body: [
      "A plain answer to whether the rules made more money than the comparison, after costs, and whether they did it safely.",
      "It shows the numbers and the checks. It does not say a strategy is good, and it is not advice.",
    ],
  },
  "excess cagr": {
    title: "Extra yearly growth",
    body: [
      "How much more the strategy grew each year than the comparison did, over the same dates, after costs.",
      "A positive number means the rules grew more.",
    ],
  },
  sharpe: {
    title: "Return for the risk taken",
    body: [
      "How much return the strategy gave for each step of ups and downs in its value.",
      "Higher is better. Look at the difference between the strategy and the comparison, not just the number on its own.",
    ],
  },
  cagr: {
    title: "Yearly growth rate",
    body: ["The steady yearly return that would give the same total over the whole period."],
  },
  volatility: {
    title: "How bumpy the ride was",
    body: ["How much the value jumped up and down, per year. A higher number means a bumpier ride."],
  },
  "max drawdown": {
    title: "Biggest fall",
    body: ["The largest drop from a high point to the next low. It shows how far down you would have been at the worst moment."],
  },
  "worst month": {
    title: "Worst month",
    body: ["The weakest calendar month in the test, as a percentage change."],
  },
  "cost drag": {
    title: "Cost of trading",
    body: ["How much yearly growth was lost to trading fees and price gaps. A large number means frequent trading is expensive."],
  },
  "in-sample": {
    title: "Earlier dates vs later dates",
    body: [
      "The test splits the history: the first 70% of dates and the last 30%. The rules were chosen without looking at the later part.",
      "If the later part is much better than the earlier part, the result may depend on recent market conditions, not the rules.",
    ],
  },
  "walk-forward": {
    title: "Consistency over time",
    body: [
      "The history is cut into six-month slices, starting after the first two years. The count shows how many slices the rules beat simply holding the same stocks.",
      "Many winning slices suggests the rules work more than once. A few suggests luck or one good period.",
    ],
  },
  "deflated sharpe": {
    title: "Could this be luck?",
    body: [
      "The chance that the result is real and not luck, after allowing for how many versions of the rules you tried in the last 90 days.",
      "The more versions you try, the more likely one of them looks good by chance, so the number goes down.",
    ],
  },
  sensitivity: {
    title: "Does a small change break it?",
    body: [
      "Each number in the rules is nudged up and down by 10%, one at a time, and the test is run again.",
      "If a small nudge changes the result a lot, the rules depend on one exact number and may not hold up.",
    ],
  },
  churn: {
    title: "Buying back too soon",
    body: [
      "The share of sales that were followed by buying the same stock again within two days.",
      "A high share means the rules are flipping in and out, which costs money.",
    ],
  },
  trades: {
    title: "Number of trades",
    body: ["One trade is one buy followed by its sell. Fewer than 30 trades is too few to judge the rules."],
  },
  "win rate": {
    title: "Winning trades",
    body: ["The share of trades that made money after costs."],
  },
  exposure: {
    title: "Time in the market",
    body: ["The share of trading days when the money was invested in a stock."],
  },
  contribution: {
    title: "Which stocks made the money",
    body: ["How much each stock added to or took from the total. If one stock makes most of the gain, the result depends on it."],
  },
  "data notes": {
    title: "Limits of this test",
    body: [
      "The stocks are the ones you chose, the stored company scores only start in August 2026, and the app's own model portfolio has only a few months of history.",
    ],
  },
  "saved strategies": {
    title: "Saved tests",
    body: ["Tests you saved, with their rules and results as they were. You can compare up to four, or share one with a read-only link."],
  },
};
