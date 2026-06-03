// @ts-nocheck
'use client'

import { KBarSearchProvider } from 'pliny/search/KBar'
import { useRouter } from 'next/navigation'
import { formatDate } from 'pliny/utils/formatDate'

export default function SearchWrapper({ searchConfig, children }) {
  const router = useRouter()
  const { kbarConfig } = searchConfig

  return (
    <KBarSearchProvider
      kbarConfig={{
        ...kbarConfig,
        onSearchDocumentsLoad: (json: any[]) => {
          const actions = []
          for (const post of json) {
            actions.push({
              id: post.path,
              name: post.title,
              keywords: post.summary || '',
              section: 'Content',
              subtitle: formatDate(post.date, 'en-US'),
              perform: () => {
                if (post.layout === 'project') {
                  window.open(post.path, '_blank', 'noopener,noreferrer')
                } else {
                  router.push('/' + post.path)
                }
              },
            })
          }
          return actions
        },
      }}
    >
      {children}
    </KBarSearchProvider>
  )
}
