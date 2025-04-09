import type { Config } from "tailwindcss"
const config: Config = {
    darkMode: "class",
    content: [
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
        "*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                border: "hsl(var(--border))",
                input: "hsl(var(--input))",
                ring: "hsl(var(--ring))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",
                primary: {
                    DEFAULT: "#10a37f",
                    foreground: "hsl(var(--primary-foreground))",
                },
                secondary: {
                    DEFAULT: "hsl(var(--secondary))",
                    foreground: "hsl(var(--secondary-foreground))",
                },
                destructive: {
                    DEFAULT: "hsl(var(--destructive))",
                    foreground: "hsl(var(--destructive-foreground))",
                },
                muted: {
                    DEFAULT: "hsl(var(--muted))",
                    foreground: "hsl(var(--muted-foreground))",
                },
                accent: {
                    DEFAULT: "hsl(var(--accent))",
                    foreground: "hsl(var(--accent-foreground))",
                },
                popover: {
                    DEFAULT: "hsl(var(--popover))",
                    foreground: "hsl(var(--popover-foreground))",
                },
                card: {
                    DEFAULT: "hsl(var(--card))",
                    foreground: "hsl(var(--card-foreground))",
                },
                "primary-hover": "#0e8e70",
                "primary-light": "#e6f7f4",
                "sidebar-bg": "#202123",
                "sidebar-hover": "#2d2d3f",
                "chat-bg": "#f7f7f8",
                "message-user-bg": "#dcf8f6",
                "message-ai-bg": "#ffffff",
                "border-color": "#e5e7eb",
                "text-primary": "#111827",
                "text-secondary": "#6b7280",
                "text-light": "#f9fafb",
            },
            borderRadius: {
                lg: "0.75rem",
                md: "0.5rem",
                sm: "0.375rem",
            },
            boxShadow: {
                sm: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                md: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                lg: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
            },
            animation: {
                pulse: "pulse 1.5s infinite",
            },
            keyframes: {
                pulse: {
                    "0%": { transform: "scale(0.95)", opacity: "0.7" },
                    "50%": { transform: "scale(1.05)", opacity: "0.3" },
                    "100%": { transform: "scale(0.95)", opacity: "0.7" },
                },
            },
        },
    },
    plugins: [require("tailwindcss-animate")],
}
export default config

