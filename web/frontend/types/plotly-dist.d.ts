// The prebuilt Plotly bundle (used by react-plotly.js) has no types of its own; reuse the plotly.js ones.
declare module "plotly.js/dist/plotly" {
  export * from "plotly.js";
  export { default } from "plotly.js";
}
