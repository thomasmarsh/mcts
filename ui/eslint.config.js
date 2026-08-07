import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    },
  },
  ...tseslint.configs.recommended,
);