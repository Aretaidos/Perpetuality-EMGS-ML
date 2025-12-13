import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { EMGDataPoint, EMG_CHANNEL_LABELS, CHANNEL_COLORS } from '../../types/gestures'

interface EMGWaveformProps {
  data: EMGDataPoint[]
}

export default function EMGWaveform({ data }: EMGWaveformProps) {
  const width = 800
  const height = 280
  const padding = { top: 20, right: 20, bottom: 30, left: 60 }
  const channelHeight = (height - padding.top - padding.bottom) / 7

  // Generate SVG paths for each channel
  const paths = useMemo(() => {
    if (data.length < 2) return []

    return Array.from({ length: 7 }, (_, channelIndex) => {
      const points = data.map((point, i) => {
        const x = padding.left + (i / (data.length - 1)) * (width - padding.left - padding.right)
        const baseY = padding.top + channelIndex * channelHeight + channelHeight / 2
        const value = point.channels[channelIndex] || 0
        const normalizedValue = (value / 150) * (channelHeight / 2) // Normalize to fit in channel height
        const y = baseY - normalizedValue
        return `${x},${y}`
      })

      return {
        path: `M ${points.join(' L ')}`,
        color: CHANNEL_COLORS[channelIndex],
        label: EMG_CHANNEL_LABELS[channelIndex],
      }
    })
  }, [data])

  // Generate placeholder paths when no data
  const placeholderPaths = useMemo(() => {
    return Array.from({ length: 7 }, (_, channelIndex) => {
      const baseY = padding.top + channelIndex * channelHeight + channelHeight / 2
      return {
        path: `M ${padding.left},${baseY} L ${width - padding.right},${baseY}`,
        color: CHANNEL_COLORS[channelIndex],
        label: EMG_CHANNEL_LABELS[channelIndex],
      }
    })
  }, [])

  const displayPaths = data.length >= 2 ? paths : placeholderPaths

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        style={{ maxHeight: '280px' }}
      >
        <defs>
          {/* Gradient backgrounds for each channel */}
          {CHANNEL_COLORS.map((color, i) => (
            <linearGradient key={i} id={`channelGradient${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.2" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          ))}
          {/* Glow effect for lines */}
          <filter id="lineGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background grid */}
        <g className="opacity-30">
          {/* Horizontal lines */}
          {Array.from({ length: 8 }, (_, i) => (
            <line
              key={`h${i}`}
              x1={padding.left}
              y1={padding.top + i * channelHeight}
              x2={width - padding.right}
              y2={padding.top + i * channelHeight}
              stroke="#CBD5E1"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
          ))}
          {/* Vertical lines */}
          {Array.from({ length: 11 }, (_, i) => (
            <line
              key={`v${i}`}
              x1={padding.left + (i / 10) * (width - padding.left - padding.right)}
              y1={padding.top}
              x2={padding.left + (i / 10) * (width - padding.left - padding.right)}
              y2={height - padding.bottom}
              stroke="#CBD5E1"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
          ))}
        </g>

        {/* Channel labels */}
        {displayPaths.map((channel, i) => (
          <text
            key={`label${i}`}
            x={padding.left - 8}
            y={padding.top + i * channelHeight + channelHeight / 2}
            textAnchor="end"
            alignmentBaseline="middle"
            fill={channel.color}
            fontSize="10"
            fontWeight="500"
            fontFamily="JetBrains Mono, monospace"
          >
            {channel.label.split(' ')[0]}
          </text>
        ))}

        {/* Waveform paths */}
        {displayPaths.map((channel, i) => (
          <motion.g key={i}>
            {/* Shadow/glow path */}
            <motion.path
              d={channel.path}
              fill="none"
              stroke={channel.color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.3"
              filter="url(#lineGlow)"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
            {/* Main path */}
            <motion.path
              d={channel.path}
              fill="none"
              stroke={channel.color}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </motion.g>
        ))}

        {/* Time axis */}
        <g>
          <line
            x1={padding.left}
            y1={height - padding.bottom}
            x2={width - padding.right}
            y2={height - padding.bottom}
            stroke="#94A3B8"
            strokeWidth="1"
          />
          {/* Time labels */}
          {['0s', '2.5s', '5s', '7.5s', '10s'].map((label, i) => (
            <text
              key={label}
              x={padding.left + (i / 4) * (width - padding.left - padding.right)}
              y={height - 8}
              textAnchor="middle"
              fill="#94A3B8"
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
            >
              {label}
            </text>
          ))}
        </g>

        {/* No data overlay */}
        {data.length < 2 && (
          <g>
            <rect
              x={padding.left}
              y={padding.top}
              width={width - padding.left - padding.right}
              height={height - padding.top - padding.bottom}
              fill="white"
              fillOpacity="0.8"
            />
            <text
              x={width / 2}
              y={height / 2}
              textAnchor="middle"
              fill="#94A3B8"
              fontSize="14"
              fontWeight="500"
            >
              Start demo to see EMG signals
            </text>
          </g>
        )}
      </svg>

      {/* Channel Legend */}
      <div className="flex flex-wrap justify-center gap-4 mt-4">
        {EMG_CHANNEL_LABELS.map((label, i) => (
          <div key={label} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CHANNEL_COLORS[i] }}
            />
            <span className="text-xs text-gray-500 font-mono">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
