'use client'

import { useEffect, useState } from 'react'
import { useTheme } from 'next-themes'
import { Sun, Moon } from 'lucide-react'
import { motion } from 'motion/react'

const ThemeSwitch = () => {
  const [mounted, setMounted] = useState(false)
  const { resolvedTheme, setTheme } = useTheme()

  useEffect(() => setMounted(true), [])

  if (!mounted) {
    return <div className="h-6 w-6" />
  }

  return (
    <button
      aria-label="Toggle dark mode"
      onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
      className="hover:text-primary-500 dark:hover:text-primary-400"
    >
      <motion.div
        key={resolvedTheme}
        initial={{ rotate: -90, opacity: 0, scale: 0.5 }}
        animate={{ rotate: 0, opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
      >
        {resolvedTheme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
      </motion.div>
    </button>
  )
}

export default ThemeSwitch
