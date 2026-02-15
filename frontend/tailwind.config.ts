import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0d1b2a",
        panel: "#f4f1de",
        accent: "#bc6c25",
        calm: "#3d5a80",
        good: "#2a9d8f",
        bad: "#e63946",
      },
      boxShadow: {
        soft: "0 10px 30px rgba(13, 27, 42, 0.12)",
      },
    },
  },
  plugins: [],
} satisfies Config;
