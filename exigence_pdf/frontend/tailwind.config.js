/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef4ff',
          100: '#d9e6ff',
          200: '#bcd3ff',
          300: '#8eb6ff',
          400: '#598eff',
          500: '#3366ff',
          600: '#1f47f5',
          700: '#1a35e1',
          800: '#1c2fb6',
          900: '#1d2f8f',
        },
        ink: {
          950: '#070b14',
          900: '#0b1120',
          800: '#111a2e',
          700: '#1b2740',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        card: '0 18px 45px -25px rgba(15, 23, 42, 0.45)',
        glow: '0 0 0 1px rgba(51, 102, 255, 0.35), 0 12px 40px -18px rgba(51, 102, 255, 0.65)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      animation: {
        'fade-up': 'fade-up 320ms ease-out both',
        shimmer: 'shimmer 1.4s linear infinite',
      },
    },
  },
  plugins: [],
}
