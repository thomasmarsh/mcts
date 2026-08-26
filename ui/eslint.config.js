import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";

export default tseslint.config(
  {
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  ...tseslint.configs.recommended,
  eslintConfigPrettier,
  // Hard rule: only api-client.ts may call fetch/XMLHttpRequest directly;
  // every reducer/component routes network calls through env.xxx().
  {
    files: ["app/src/**/*.{ts,tsx}", "packages/*/src/**/*.{ts,tsx}"],
    ignores: ["**/api-client.ts"],
    rules: {
      "no-restricted-globals": [
        "error",
        {
          name: "fetch",
          message: "Only api-client.ts may call fetch -- route network calls through env.xxx().",
        },
        {
          name: "XMLHttpRequest",
          message:
            "Only api-client.ts may talk to the network directly -- route network calls through env.xxx().",
        },
      ],
    },
  },
);
