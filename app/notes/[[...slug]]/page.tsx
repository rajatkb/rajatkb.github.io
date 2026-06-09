import 'css/prism.css'

import PageTitle from '@/components/PageTitle'
import { MDXLayoutRenderer } from 'pliny/mdx-components'
import { allVaultNotes } from 'contentlayer/generated'
import type { VaultNote } from 'contentlayer/generated'
import { Metadata } from 'next'
import siteMetadata from '@/data/siteMetadata'
import { notFound } from 'next/navigation'
import { formatDate } from 'pliny/utils/formatDate'
import SectionContainer from '@/components/SectionContainer'
import Link from '@/components/Link'
import Tag from '@/components/Tag'

// ─── Helpers ─────────────────────────────────────────────────────────

function findNote(slug: string): VaultNote | undefined {
  return allVaultNotes.find((p) => p.slug === slug)
}

/** Check if a path is a directory (has children with this prefix) */
function isDirectory(path: string): boolean {
  const prefix = path ? path + '/' : ''
  return allVaultNotes.some((n) => n.slug.startsWith(prefix) && n.slug !== path)
}

/** Get direct children of a folder path ('' for root) */
function getChildren(folderPath: string) {
  const prefix = folderPath ? folderPath + '/' : ''
  const dirs = new Map<string, { date: string; count: number }>()
  const files: VaultNote[] = []

  for (const note of allVaultNotes) {
    if (note.draft === true) continue
    if (!note.slug.startsWith(prefix)) continue

    const remainder = note.slug.slice(prefix.length)
    const nextSlash = remainder.indexOf('/')

    if (nextSlash === -1) {
      files.push(note)
    } else {
      const dirName = remainder.slice(0, nextSlash)
      const existing = dirs.get(dirName)
      if (!existing || note.date > existing.date) {
        dirs.set(dirName, { date: note.date, count: (existing?.count || 0) + 1 })
      } else {
        dirs.set(dirName, { date: existing.date, count: (existing?.count || 0) + 1 })
      }
    }
  }

  return {
    dirs: [...dirs.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([name, meta]) => ({ name, date: meta.date, count: meta.count })),
    files: files.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()),
  }
}

/** Derive all folder paths (for generateStaticParams) */
function allFolderPaths(): string[] {
  const folders = new Set<string>()
  for (const note of allVaultNotes) {
    const parts = note.slug.split('/')
    // Every path prefix is a folder
    for (let i = 1; i < parts.length; i++) {
      folders.add(parts.slice(0, i).join('/'))
    }
  }
  return [...folders]
}

/** Display name from a slug segment */
function displayName(segment: string): string {
  return segment
    .replace(/^\d+-/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

// ─── Folder Listing ───────────────────────────────────────────────────

function FolderListing({ folderPath }: { folderPath: string }) {
  const { dirs, files } = getChildren(folderPath)

  // Build the display name: show as breadcrumb-style path or just the last segment
  const title = folderPath
    ? folderPath
        .split('/')
        .map(displayName)
        .join(' / ')
    : 'Notes'

  return (
    <SectionContainer>
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="space-y-2 pt-6 pb-8 md:space-y-5">
          <h1 className="text-3xl leading-9 font-extrabold tracking-tight text-gray-900 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14 dark:text-gray-100">
            {title}
          </h1>
          {folderPath === '' && (
            <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
              Personal notes, research snippets, and reference material from my Obsidian vault.
            </p>
          )}
        </div>
        <ul>
          {dirs.map((dir) => {
            const childPath = folderPath ? `${folderPath}/${dir.name}` : dir.name
            return (
              <li key={childPath} className="py-5">
                <article className="flex flex-col space-y-2 xl:space-y-0">
                  <dl>
                    <dt className="sr-only">Published on</dt>
                    <dd className="text-base leading-6 font-medium text-gray-500 dark:text-gray-400">
                      <time dateTime={dir.date} suppressHydrationWarning>
                        {formatDate(dir.date, siteMetadata.locale)}
                      </time>
                    </dd>
                  </dl>
                  <div className="space-y-3">
                    <div>
                      <h2 className="text-2xl leading-8 font-bold tracking-tight">
                        <Link
                          href={`/notes/${childPath}`}
                          className="inline-flex items-center gap-2 text-gray-900 dark:text-gray-100"
                        >
                          {displayName(dir.name)}
                          <span className="text-sm text-gray-400 dark:text-gray-500">/</span>
                        </Link>
                      </h2>
                    </div>
                    {dir.count > 0 && (
                      <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                        {dir.count} {dir.count === 1 ? 'note' : 'notes'}
                      </div>
                    )}
                  </div>
                </article>
              </li>
            )
          })}
          {files.map((note) => (
            <li key={note.slug} className="py-5">
              <article className="flex flex-col space-y-2 xl:space-y-0">
                <dl>
                  <dt className="sr-only">Published on</dt>
                  <dd className="text-base leading-6 font-medium text-gray-500 dark:text-gray-400">
                    <time dateTime={note.date} suppressHydrationWarning>
                      {formatDate(note.date, siteMetadata.locale)}
                    </time>
                  </dd>
                </dl>
                <div className="space-y-3">
                  <div>
                    <h2 className="text-2xl leading-8 font-bold tracking-tight">
                      <Link
                        href={`/notes/${note.slug}`}
                        className="text-gray-900 dark:text-gray-100"
                      >
                        {note.title}
                      </Link>
                    </h2>
                    {note.tags && note.tags.length > 0 && (
                      <div className="flex flex-wrap">
                        {note.tags.map((tag: string) => (
                          <Tag key={tag} text={tag} />
                        ))}
                      </div>
                    )}
                  </div>
                  {note.summary && (
                    <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                      {note.summary}
                    </div>
                  )}
                </div>
              </article>
            </li>
          ))}
          {dirs.length === 0 && files.length === 0 && (
            <p className="py-8 text-gray-500 dark:text-gray-400">No notes here yet.</p>
          )}
        </ul>
      </div>
    </SectionContainer>
  )
}

// ─── Metadata ─────────────────────────────────────────────────────────

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>
}): Promise<Metadata | undefined> {
  const params = await props.params
  const slugPath = params.slug?.join('/') || ''

  // Folder listing page
  if (!slugPath || isDirectory(slugPath)) {
    const title = slugPath
      ? slugPath.split('/').map(displayName).join(' / ')
      : 'Notes'
    return {
      title,
      description: slugPath
        ? `Notes in ${title}`
        : 'Personal notes, research snippets, and reference material.',
    }
  }

  // Note page
  const note = findNote(slugPath)
  if (!note) return

  return {
    title: note.title,
    description: note.summary,
    openGraph: {
      title: note.title,
      description: note.summary,
      siteName: siteMetadata.title,
      locale: 'en_US',
      type: 'article',
      publishedTime: new Date(note.date).toISOString(),
      url: './',
    },
    twitter: {
      card: 'summary_large_image',
      title: note.title,
      description: note.summary,
    },
  }
}

// ─── Static Params ────────────────────────────────────────────────────

export const generateStaticParams = async () => {
  // Note pages
  const noteParams = allVaultNotes.map((p) => ({
    slug: p.slug.split('/').map((name) => decodeURI(name)),
  }))
  // Folder listing pages
  const folderParams = allFolderPaths().map((f) => ({
    slug: f.split('/').map((name) => decodeURI(name)),
  }))
  // Root /notes/ page
  const rootParam = { slug: undefined }
  return [rootParam, ...noteParams, ...folderParams]
}

// ─── Page ─────────────────────────────────────────────────────────────

export default async function NotesPage(props: {
  params: Promise<{ slug?: string[] }>
}) {
  const params = await props.params
  const slugPath = params.slug?.join('/') || ''

  // Case 1: root or folder — show directory listing
  if (!slugPath || isDirectory(slugPath)) {
    return <FolderListing folderPath={slugPath} />
  }

  // Case 2: note page
  const note = findNote(slugPath) as VaultNote | undefined
  if (!note) return notFound()

  return (
    <SectionContainer>
      <article>
        <div className="space-y-1 border-b border-gray-200 pb-10 text-center dark:border-gray-700">
          <dl>
            <div>
              <dt className="sr-only">Published on</dt>
              <dd className="text-base leading-6 font-medium text-gray-500 dark:text-gray-400">
                <time dateTime={note.date}>{formatDate(note.date, siteMetadata.locale)}</time>
              </dd>
            </div>
          </dl>
          <PageTitle>{note.title}</PageTitle>
        </div>
        <div className="divide-y divide-gray-200 pb-8 dark:divide-gray-700">
          <div className="prose dark:prose-invert max-w-none pt-10 pb-8">
            <MDXLayoutRenderer code={note.body.code} />
          </div>
        </div>
      </article>
    </SectionContainer>
  )
}
