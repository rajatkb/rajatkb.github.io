import { allVaultNotes } from 'contentlayer/generated'
import type { VaultNote } from 'contentlayer/generated'
import type { Metadata } from 'next'
import { genPageMetadata } from 'app/seo'
import { formatDate } from 'pliny/utils/formatDate'
import siteMetadata from '@/data/siteMetadata'
import Link from '@/components/Link'
import Tag from '@/components/Tag'

export const metadata: Metadata = genPageMetadata({ title: 'Notes' })

function getNotes(): VaultNote[] {
  return allVaultNotes
    .filter((note) => note.draft !== true)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
}

export default async function NotesPage() {
  const notes = getNotes()

  return (
    <div className="divide-y divide-gray-200 dark:divide-gray-700">
      <div className="space-y-2 pt-6 pb-8 md:space-y-5">
        <h1 className="text-3xl leading-9 font-extrabold tracking-tight text-gray-900 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14 dark:text-gray-100">
          Notes
        </h1>
        <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
          Personal notes, research snippets, and reference material from my Obsidian vault.
        </p>
      </div>
      <ul>
        {notes.map((note) => {
          const { path, date, title, summary, tags } = note
          return (
            <li key={path} className="py-5">
              <article className="flex flex-col space-y-2 xl:space-y-0">
                <dl>
                  <dt className="sr-only">Published on</dt>
                  <dd className="text-base leading-6 font-medium text-gray-500 dark:text-gray-400">
                    <time dateTime={date} suppressHydrationWarning>
                      {formatDate(date, siteMetadata.locale)}
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
                        {title}
                      </Link>
                    </h2>
                    {tags && tags.length > 0 && (
                      <div className="flex flex-wrap">
                        {tags.map((tag) => (
                          <Tag key={tag} text={tag} />
                        ))}
                      </div>
                    )}
                  </div>
                  {summary && (
                    <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                      {summary}
                    </div>
                  )}
                </div>
              </article>
            </li>
          )
        })}
      </ul>
      {notes.length === 0 && <p className="py-8 text-gray-500 dark:text-gray-400">No notes yet.</p>}
    </div>
  )
}
