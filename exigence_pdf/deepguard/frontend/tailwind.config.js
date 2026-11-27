/** DeepGuard design system — merged tokens:
 *  - my flat tokens (bg/panel/line/ink/mute/faint via CSS vars) for the platform pages
 *  - the ORIGINAL repo frontend palette (brand 50-900, ink 700-950) for the /analyze
 *    output page ported verbatim from exigence_pdf/frontend (darkMode: 'class')
 *  - fonts Inter + JetBrains Mono shared by both systems  */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--bg-rgb) / <alpha-value>)',
        panel: 'rgb(var(--panel-rgb) / <alpha-value>)',
        panel2: 'rgb(var(--panel2-rgb) / <alpha-value>)',
        line: 'rgb(var(--line-rgb) / <alpha-value>)',
        ink: {
          DEFAULT: 'rgb(var(--ink-rgb) / <alpha-value>)',
          700: '#1b2740',
          800: '#111a2e',
          900: '#0b1120',
          950: '#070b14',
        },
        mute: 'rgb(var(--mute-rgb) / <alpha-value>)',
        faint: 'rgb(var(--faint-rgb) / <alpha-value>)',
        brand: {
          DEFAULT: '#3b6bff',
          deep: '#1f47f5',
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
        ok: '#10b981',
        bad: '#f43f5e',
        warn: '#f59e0b',
        violet: {
          DEFAULT: '#8b5cf6',
          50: '#f5f3ff', 100: '#ede9fe', 200: '#ddd6fe', 300: '#c4b5fd',
          400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed', 700: '#6d28d9',
          800: '#5b21b6', 900: '#4c1d95', 950: '#2e1065',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 0 rgba(255,255,255,.04) inset, 0 8px 24px -12px rgba(0,0,0,.5)',
        pop: '0 12px 40px -8px rgba(0,0,0,.55)',
        glow: '0 0 0 1px rgba(59,107,255,.35), 0 8px 32px -8px rgba(59,107,255,.45)',
        'daf-card': '0 18px 45px -25px rgba(15, 23, 42, 0.45)',
      },
      animation: {
        'fade-in': 'fadeIn .35s ease both',
        'rise': 'rise .4s ease both',
        'pulse-soft': 'pulseSoft 2.2s ease-in-out infinite',
        'scan': 'scan 1.6s ease-in-out infinite',
        'fade-up': 'fade-up 320ms ease-out both',
        'shimmer': 'shimmer 1.4s linear infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
        rise: { from: { opacity: 0, transform: 'translateY(10px)' }, to: { opacity: 1, transform: 'none' } },
        pulseSoft: { '0%,100%': { opacity: 1 }, '50%': { opacity: .55 } },
        scan: { '0%': { transform: 'translateY(-100%)' }, '100%': { transform: 'translateY(400%)' } },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
