import { motion } from 'framer-motion'
import { Play, Pause, RotateCcw, Download, Settings } from 'lucide-react'

interface FooterProps {
  demoRunning: boolean
  onToggleDemo: () => void
  onReset: () => void
}

export default function Footer({ demoRunning, onToggleDemo, onReset }: FooterProps) {
  return (
    <footer className="glass-panel border-t border-white/20 sticky bottom-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left - Demo Controls */}
          <div className="flex items-center gap-3">
            <motion.button
              onClick={onToggleDemo}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium transition-all ${
                demoRunning
                  ? 'bg-amber-100 text-amber-700 hover:bg-amber-200'
                  : 'bg-gradient-to-r from-accent-primary to-accent-secondary text-white hover:opacity-90'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {demoRunning ? (
                <>
                  <Pause className="w-4 h-4" />
                  <span>Pause Demo</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Start Demo</span>
                </>
              )}
            </motion.button>

            <motion.button
              onClick={onReset}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <RotateCcw className="w-4 h-4" />
              <span className="hidden sm:inline">Reset</span>
            </motion.button>
          </div>

          {/* Center - Branding */}
          <div className="hidden md:block text-center">
            <p className="text-xs text-gray-400">
              Powered by TinyML Neural Networks
            </p>
          </div>

          {/* Right - Additional Controls */}
          <div className="flex items-center gap-3">
            <motion.button
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Export</span>
            </motion.button>

            <motion.button
              className="flex items-center justify-center w-10 h-10 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Settings className="w-4 h-4" />
            </motion.button>
          </div>
        </div>
      </div>
    </footer>
  )
}
