import { motion } from 'framer-motion'
import { ThumbAction } from '../../types/gestures'
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react'

interface HandVisualizationProps {
  indexPressed: boolean
  middlePressed: boolean
  thumbAction: ThumbAction
  indexConfidence: number
  middleConfidence: number
  thumbConfidence: number
}

export default function HandVisualization({
  indexPressed,
  middlePressed,
  thumbAction,
  indexConfidence,
  middleConfidence,
  thumbConfidence,
}: HandVisualizationProps) {
  // Get thumb direction arrow
  const ThumbArrow = () => {
    switch (thumbAction) {
      case 'up':
        return <ArrowUp className="w-6 h-6 text-motion-up" />
      case 'down':
        return <ArrowDown className="w-6 h-6 text-motion-down" />
      case 'in':
        return <ArrowLeft className="w-6 h-6 text-motion-in" />
      case 'out':
        return <ArrowRight className="w-6 h-6 text-motion-out" />
      default:
        return null
    }
  }

  return (
    <div className="relative flex items-center justify-center py-8">
      {/* Neural Network Background Effect */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-80 h-80 rounded-full bg-gradient-to-br from-indigo-100/50 to-purple-100/30 blur-2xl" />
      </div>

      {/* Main SVG Hand */}
      <svg
        viewBox="0 0 400 500"
        className="w-full max-w-md h-auto relative z-10"
        style={{ filter: 'drop-shadow(0 10px 30px rgba(0, 0, 0, 0.1))' }}
      >
        <defs>
          {/* Gradients for active states */}
          <linearGradient id="fingerActiveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#818CF8" />
            <stop offset="100%" stopColor="#6366F1" />
          </linearGradient>
          <linearGradient id="thumbActiveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#A78BFA" />
            <stop offset="100%" stopColor="#8B5CF6" />
          </linearGradient>
          <linearGradient id="palmGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#F8FAFC" />
            <stop offset="100%" stopColor="#E2E8F0" />
          </linearGradient>

          {/* Glow filters */}
          <filter id="glowIndigo" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="8" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glowPurple" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="10" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="4" stdDeviation="6" floodColor="#000" floodOpacity="0.1" />
          </filter>
        </defs>

        {/* Palm */}
        <motion.path
          d="M 120 280
             Q 100 320, 110 380
             Q 120 440, 180 460
             Q 240 480, 300 450
             Q 340 420, 340 360
             Q 340 300, 320 260
             Q 300 220, 280 200
             L 200 200
             Q 140 210, 120 280 Z"
          fill="url(#palmGradient)"
          stroke="#CBD5E1"
          strokeWidth="2"
          filter="url(#softShadow)"
        />

        {/* Ring Finger (background) */}
        <motion.path
          d="M 270 200
             Q 275 150, 275 100
             Q 275 60, 265 50
             Q 250 40, 235 50
             Q 225 60, 225 100
             Q 225 150, 230 200 Z"
          fill="#F1F5F9"
          stroke="#CBD5E1"
          strokeWidth="2"
          filter="url(#softShadow)"
        />

        {/* Pinky Finger (background) */}
        <motion.path
          d="M 310 220
             Q 320 170, 318 130
             Q 316 100, 305 90
             Q 290 80, 278 90
             Q 268 100, 270 130
             Q 272 170, 280 220 Z"
          fill="#F1F5F9"
          stroke="#CBD5E1"
          strokeWidth="2"
          filter="url(#softShadow)"
        />

        {/* Middle Finger */}
        <motion.path
          d="M 220 200
             Q 222 130, 220 70
             Q 218 30, 200 20
             Q 182 10, 170 25
             Q 160 40, 165 70
             Q 170 130, 175 200 Z"
          fill={middlePressed ? 'url(#fingerActiveGradient)' : '#F1F5F9'}
          stroke={middlePressed ? '#6366F1' : '#CBD5E1'}
          strokeWidth={middlePressed ? 3 : 2}
          filter={middlePressed ? 'url(#glowIndigo)' : 'url(#softShadow)'}
          animate={{
            scale: middlePressed ? 1.02 : 1,
          }}
          transition={{ duration: 0.2 }}
          style={{ transformOrigin: '195px 110px' }}
        />

        {/* Index Finger */}
        <motion.path
          d="M 170 210
             Q 165 150, 155 90
             Q 148 50, 130 45
             Q 110 40, 100 55
             Q 90 72, 100 100
             Q 110 150, 125 210 Z"
          fill={indexPressed ? 'url(#fingerActiveGradient)' : '#F1F5F9'}
          stroke={indexPressed ? '#6366F1' : '#CBD5E1'}
          strokeWidth={indexPressed ? 3 : 2}
          filter={indexPressed ? 'url(#glowIndigo)' : 'url(#softShadow)'}
          animate={{
            scale: indexPressed ? 1.02 : 1,
          }}
          transition={{ duration: 0.2 }}
          style={{ transformOrigin: '135px 130px' }}
        />

        {/* Thumb */}
        <motion.path
          d="M 110 280
             Q 80 260, 55 240
             Q 30 220, 25 200
             Q 22 175, 40 165
             Q 60 155, 80 170
             Q 100 185, 120 210
             Q 130 230, 120 260 Z"
          fill={thumbAction !== 'idle' ? 'url(#thumbActiveGradient)' : '#F1F5F9'}
          stroke={thumbAction !== 'idle' ? '#8B5CF6' : '#CBD5E1'}
          strokeWidth={thumbAction !== 'idle' ? 3 : 2}
          filter={thumbAction !== 'idle' ? 'url(#glowPurple)' : 'url(#softShadow)'}
          animate={{
            scale: thumbAction === 'click' ? [1, 1.1, 1] : 1,
          }}
          transition={{ duration: 0.3 }}
          style={{ transformOrigin: '70px 215px' }}
        />

        {/* Finger Joints (decorative) */}
        <circle cx="200" cy="60" r="4" fill="#CBD5E1" opacity={middlePressed ? 0 : 0.5} />
        <circle cx="130" cy="80" r="4" fill="#CBD5E1" opacity={indexPressed ? 0 : 0.5} />
        <circle cx="55" cy="200" r="4" fill="#CBD5E1" opacity={thumbAction !== 'idle' ? 0 : 0.5} />

        {/* Neural Connection Lines (when active) */}
        {indexPressed && (
          <motion.path
            d="M 125 210 Q 150 350, 200 400"
            fill="none"
            stroke="#6366F1"
            strokeWidth="2"
            strokeDasharray="6 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.5 }}
            transition={{ duration: 0.5 }}
          />
        )}
        {middlePressed && (
          <motion.path
            d="M 195 200 Q 200 330, 220 400"
            fill="none"
            stroke="#6366F1"
            strokeWidth="2"
            strokeDasharray="6 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.5 }}
            transition={{ duration: 0.5 }}
          />
        )}
        {thumbAction !== 'idle' && (
          <motion.path
            d="M 80 240 Q 140 320, 180 400"
            fill="none"
            stroke="#8B5CF6"
            strokeWidth="2"
            strokeDasharray="6 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.5 }}
            transition={{ duration: 0.5 }}
          />
        )}

        {/* Confidence Labels */}
        {indexPressed && (
          <motion.g initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <rect x="55" y="35" width="50" height="22" rx="6" fill="#6366F1" />
            <text x="80" y="51" textAnchor="middle" fill="white" fontSize="12" fontWeight="600">
              {indexConfidence.toFixed(0)}%
            </text>
          </motion.g>
        )}
        {middlePressed && (
          <motion.g initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <rect x="175" y="0" width="50" height="22" rx="6" fill="#6366F1" />
            <text x="200" y="16" textAnchor="middle" fill="white" fontSize="12" fontWeight="600">
              {middleConfidence.toFixed(0)}%
            </text>
          </motion.g>
        )}
        {thumbAction !== 'idle' && (
          <motion.g initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <rect x="0" y="145" width="50" height="22" rx="6" fill="#8B5CF6" />
            <text x="25" y="161" textAnchor="middle" fill="white" fontSize="12" fontWeight="600">
              {thumbConfidence.toFixed(0)}%
            </text>
          </motion.g>
        )}
      </svg>

      {/* Thumb Direction Indicator */}
      {thumbAction !== 'idle' && thumbAction !== 'click' && (
        <motion.div
          className="absolute left-16 top-1/2 transform -translate-y-1/2"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.5 }}
        >
          <div className="glass-panel rounded-full p-3">
            <ThumbArrow />
          </div>
        </motion.div>
      )}

      {/* Click Ripple Effect */}
      {thumbAction === 'click' && (
        <motion.div
          className="absolute left-24 top-48"
          initial={{ scale: 0.5, opacity: 1 }}
          animate={{ scale: 2, opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="w-16 h-16 rounded-full border-4 border-purple-400" />
        </motion.div>
      )}

      {/* Status Labels */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-center gap-8 pb-4">
        <motion.div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
            indexPressed
              ? 'bg-indigo-100 text-indigo-700'
              : 'bg-gray-100 text-gray-400'
          }`}
          animate={{ scale: indexPressed ? 1.05 : 1 }}
        >
          <div className={`w-2 h-2 rounded-full ${indexPressed ? 'bg-indigo-500' : 'bg-gray-300'}`} />
          Index
        </motion.div>
        <motion.div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
            middlePressed
              ? 'bg-indigo-100 text-indigo-700'
              : 'bg-gray-100 text-gray-400'
          }`}
          animate={{ scale: middlePressed ? 1.05 : 1 }}
        >
          <div className={`w-2 h-2 rounded-full ${middlePressed ? 'bg-indigo-500' : 'bg-gray-300'}`} />
          Middle
        </motion.div>
        <motion.div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
            thumbAction !== 'idle'
              ? 'bg-purple-100 text-purple-700'
              : 'bg-gray-100 text-gray-400'
          }`}
          animate={{ scale: thumbAction !== 'idle' ? 1.05 : 1 }}
        >
          <div className={`w-2 h-2 rounded-full ${thumbAction !== 'idle' ? 'bg-purple-500' : 'bg-gray-300'}`} />
          Thumb {thumbAction !== 'idle' && `(${thumbAction})`}
        </motion.div>
      </div>
    </div>
  )
}
