/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Background colors
        'bg-primary': '#FAFBFC',
        'bg-secondary': '#F0F4F8',
        'bg-tertiary': '#E2E8F0',

        // Accent colors (Neural/AI Theme)
        'accent-primary': '#6366F1',
        'accent-secondary': '#8B5CF6',
        'accent-tertiary': '#06B6D4',

        // Finger states
        'finger-idle': '#E2E8F0',
        'finger-active': '#6366F1',
        'thumb-active': '#8B5CF6',

        // Motion indicators
        'motion-up': '#10B981',
        'motion-down': '#F59E0B',
        'motion-in': '#06B6D4',
        'motion-out': '#EC4899',

        // Status
        'status-connected': '#10B981',
        'status-disconnected': '#EF4444',
        'status-processing': '#F59E0B',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Space Grotesk', 'sans-serif'],
      },
      boxShadow: {
        'glow-indigo': '0 0 20px rgba(99, 102, 241, 0.4)',
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.4)',
        'glow-cyan': '0 0 20px rgba(6, 182, 212, 0.4)',
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.08)',
      },
      backdropBlur: {
        'glass': '12px',
      },
      animation: {
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'finger-press': 'fingerPress 0.3s ease-out forwards',
        'thumb-pulse': 'thumbPulse 0.4s ease-out forwards',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        fingerPress: {
          '0%': { transform: 'scale(1)', filter: 'drop-shadow(0 0 0 transparent)' },
          '50%': { transform: 'scale(1.08)', filter: 'drop-shadow(0 0 20px rgba(99, 102, 241, 0.6))' },
          '100%': { transform: 'scale(1.05)', filter: 'drop-shadow(0 0 12px rgba(99, 102, 241, 0.4))' },
        },
        thumbPulse: {
          '0%': { transform: 'scale(1)', filter: 'drop-shadow(0 0 0 transparent)' },
          '25%': { transform: 'scale(1.15)', filter: 'drop-shadow(0 0 30px rgba(139, 92, 246, 0.6))' },
          '100%': { transform: 'scale(1)', filter: 'drop-shadow(0 0 0 transparent)' },
        },
      },
    },
  },
  plugins: [],
}
