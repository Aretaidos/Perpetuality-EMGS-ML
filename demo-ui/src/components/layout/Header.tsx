import { motion } from 'framer-motion'
import { Brain, Wifi, WifiOff, Activity } from 'lucide-react'

interface HeaderProps {
  connectionStatus: 'connected' | 'disconnected' | 'processing'
  samplesProcessed: number
}

export default function Header({ connectionStatus, samplesProcessed }: HeaderProps) {
  return (
    <header className="glass-panel border-b border-white/20 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <motion.div
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Brain className="w-6 h-6 text-white" />
            </motion.div>
            <div>
              <h1 className="text-xl font-bold gradient-text">PERPETUALITY</h1>
              <p className="text-xs text-gray-500 -mt-1">Neural Interface Demo</p>
            </div>
          </div>

          {/* Center - Status */}
          <div className="hidden md:flex items-center gap-6">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/50">
              <Activity className="w-4 h-4 text-accent-primary" />
              <span className="text-sm font-medium text-gray-700">
                {samplesProcessed.toLocaleString()} samples
              </span>
            </div>
          </div>

          {/* Right - Connection Status */}
          <div className="flex items-center gap-4">
            <motion.div
              className={`flex items-center gap-2 px-4 py-2 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-emerald-50 text-emerald-600'
                  : connectionStatus === 'processing'
                  ? 'bg-amber-50 text-amber-600'
                  : 'bg-red-50 text-red-500'
              }`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              {connectionStatus === 'connected' || connectionStatus === 'processing' ? (
                <Wifi className="w-4 h-4" />
              ) : (
                <WifiOff className="w-4 h-4" />
              )}
              <span className="text-sm font-medium capitalize">{connectionStatus}</span>
              {connectionStatus === 'processing' && (
                <motion.div
                  className="w-2 h-2 rounded-full bg-amber-500"
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </header>
  )
}
