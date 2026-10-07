import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import jsxA11y from "eslint-plugin-jsx-a11y";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // NFR-7: eslint-config-next's core-web-vitals only cherry-picks a
  // handful of jsx-a11y rules (alt-text, aria-props/proptypes, aria-
  // unsupported-elements, role-has-required-aria-props/supports-aria-
  // props) -- missing exactly the rules that catch the most common real
  // bug (a <div onClick> with no keyboard affordance or accessible
  // role), so the full recommended set's rules are added explicitly.
  // Just `.rules`, not the whole flatConfigs.recommended object -- that
  // object also redeclares the "jsx-a11y" plugin itself, which
  // eslint-config-next already registered; flat config rejects a second
  // "plugins" entry under the same name even for the identical package.
  { rules: jsxA11y.flatConfigs.recommended.rules },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
