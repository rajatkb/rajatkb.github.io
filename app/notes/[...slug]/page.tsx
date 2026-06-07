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

export async function generateMetadata(props: {
  params: Promise<{ slug: string[] }>
}): Promise<Metadata | undefined> {
  const params = await props.params
  const slug = decodeURI(params.slug.join('/'))
  const note = allVaultNotes.find((p) => p.slug === slug)
  if (!note) return

  const publishedAt = new Date(note.date).toISOString()

  return {
    title: note.title,
    description: note.summary,
    openGraph: {
      title: note.title,
      description: note.summary,
      siteName: siteMetadata.title,
      locale: 'en_US',
      type: 'article',
      publishedTime: publishedAt,
      url: './',
    },
    twitter: {
      card: 'summary_large_image',
      title: note.title,
      description: note.summary,
    },
  }
}

export const generateStaticParams = async () => {
  return allVaultNotes.map((p) => ({ slug: p.slug.split('/').map((name) => decodeURI(name)) }))
}

export default async function Page(props: { params: Promise<{ slug: string[] }> }) {
  const params = await props.params
  const slug = decodeURI(params.slug.join('/'))

  const note = allVaultNotes.find((p) => p.slug === slug) as VaultNote | undefined
  if (!note) {
    return notFound()
  }

  return (
    <SectionContainer>
      <article>
        <div>
          <header>
            <div className="space-y-1 border-b border-gray-200 pb-10 text-center dark:border-gray-700">
              <dl>
                <div>
                  <dt className="sr-only">Published on</dt>
                  <dd className="text-base leading-6 font-medium text-gray-500 dark:text-gray-400">
                    <time dateTime={note.date}>{formatDate(note.date, siteMetadata.locale)}</time>
                  </dd>
                </div>
              </dl>
              <div>
                <PageTitle>{note.title}</PageTitle>
              </div>
            </div>
          </header>
          <div className="divide-y divide-gray-200 pb-8 dark:divide-gray-700">
            <div className="prose dark:prose-invert max-w-none pt-10 pb-8">
              <MDXLayoutRenderer code={note.body.code} />
            </div>
          </div>
        </div>
      </article>
    </SectionContainer>
  )
}
