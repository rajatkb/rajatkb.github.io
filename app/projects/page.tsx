'use client'

import { useState, useMemo } from 'react'
import projectsData from '@/data/projectsData'
import Card from '@/components/Card'

export default function Projects() {
  const [yearFilter, setYearFilter] = useState('all')
  const [tagFilter, setTagFilter] = useState('all')

  const years = useMemo(() => {
    const y = [...new Set(projectsData.map((p) => p.year))]
    return y.sort((a, b) => b - a)
  }, [])

  const allTags = useMemo(() => {
    const t = new Set<string>()
    projectsData.forEach((p) => p.tags.forEach((tag) => t.add(tag)))
    return [...t].sort()
  }, [])

  const filtered = useMemo(() => {
    return projectsData.filter((p) => {
      if (yearFilter !== 'all' && p.year !== Number(yearFilter)) return false
      if (tagFilter !== 'all' && !p.tags.includes(tagFilter)) return false
      return true
    })
  }, [yearFilter, tagFilter])

  return (
    <>
      {/* Hero */}
      <section className="relative mb-12 overflow-hidden rounded-2xl">
        <div className="absolute inset-0">
          <img
            src="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&q=80"
            alt=""
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-black/70 to-black/30" />
        </div>
        <div className="relative z-10 px-8 py-20 sm:px-12 sm:py-28">
          <h1 className="font-serif text-4xl font-bold tracking-tight text-white sm:text-5xl md:text-6xl">
            Projects
          </h1>
          <p className="mt-4 max-w-2xl text-lg leading-relaxed text-gray-200 sm:text-xl">
            ML systems, super-resolution, databases, and distributed tooling.
          </p>
        </div>
      </section>

      {/* Filters */}
      <div className="mb-8 flex flex-wrap gap-4">
        <select
          value={yearFilter}
          onChange={(e) => setYearFilter(e.target.value)}
          className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200"
        >
          <option value="all">All Years</option>
          {years.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>

        <select
          value={tagFilter}
          onChange={(e) => setTagFilter(e.target.value)}
          className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200"
        >
          <option value="all">All Categories</option>
          {allTags.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>

        {filtered.length !== projectsData.length && (
          <button
            onClick={() => {
              setYearFilter('all')
              setTagFilter('all')
            }}
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-500 hover:text-gray-700 dark:border-gray-600 dark:text-gray-400 dark:hover:text-gray-200"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Grid */}
      <div className="container py-12">
        {filtered.length === 0 ? (
          <p className="text-center text-gray-500 dark:text-gray-400">No projects match the selected filters.</p>
        ) : (
          <div className="-m-4 flex flex-wrap">
            {filtered.map((d) => (
              <Card
                key={d.title}
                title={d.title}
                description={d.description}
                imgSrc={d.imgSrc}
                href={d.href}
              />
            ))}
          </div>
        )}
      </div>
    </>
  )
}
