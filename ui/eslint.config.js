import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    },
  },
  ...tseslint.configs.recommended,
  // PLAN-UI.md's "Hard rule": only api-client.ts may call fetch/XMLHttpRequest
  // directly; every reducer/component routes network calls through env.xxx().
  {
    files: ["app/src/**/*.{ts,tsx}", "packages/*/src/**/*.{ts,tsx}"],
    ignores: ["**/api-client.ts"],
    rules: {
      "no-restricted-globals": [
        "error",
        {
          name: "fetch",
          message: "Only api-client.ts may call fetch -- route network calls through env.xxx() (PLAN-UI.md's 'Hard rule').",
        },
        {
          name: "XMLHttpRequest",
          message: "Only api-client.ts may talk to the network directly -- route network calls through env.xxx() (PLAN-UI.md's 'Hard rule').",
        },
      ],
    },
  },
);