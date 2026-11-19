/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#09090b', // Deep rich black/zinc-950
        card: '#121214',       // Charcoal card
        'card-hover': '#18181b',
        border: '#27272a',     // Subtle zinc-800 border
        'border-focus': '#52525b',
        muted: '#18181b',
        'muted-foreground': '#a1a1aa',
        foreground: '#f4f4f5',
        primary: '#ffffff',
        'primary-foreground': '#09090b',
        secondary: '#27272a',
        'secondary-foreground': '#fafafa',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace']
      }
    },
  },
  plugins: [],
}
