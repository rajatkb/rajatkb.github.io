/** @type {import("pliny/config").PlinyConfig } */
const siteMetadata = {
  title: 'Rajat Kanti Bhattacharjee',
  author: 'Rajat Kanti Bhattacharjee',
  headerTitle: 'rajatkb',
  description: 'Building neural nets, systems, and things at the edge of curiosity.',
  language: 'en-us',
  theme: 'system', // system, dark or light
  siteUrl: 'https://rajatkb.github.io',
  siteRepo: 'https://github.com/rajatkb/rajatkb.github.io',
  siteLogo: `${process.env.BASE_PATH || ''}/static/images/logo.png`,
  socialBanner: `${process.env.BASE_PATH || ''}/static/images/twitter-card.png`,
  email: 'rajatk.dev@gmail.com',
  github: 'https://github.com/rajatkb',
  x: 'https://x.com/rajat_kb',
  linkedin: 'https://www.linkedin.com/in/rajatkb',
  locale: 'en-US',
  stickyNav: false,
  analytics: {},
  comments: process.env.NEXT_PUBLIC_GISCUS_REPO
    ? {
        provider: 'giscus',
        giscusConfig: {
          repo: process.env.NEXT_PUBLIC_GISCUS_REPO,
          repositoryId: process.env.NEXT_PUBLIC_GISCUS_REPOSITORY_ID,
          category: process.env.NEXT_PUBLIC_GISCUS_CATEGORY,
          categoryId: process.env.NEXT_PUBLIC_GISCUS_CATEGORY_ID,
          mapping: 'pathname',
          reactions: '1',
          metadata: '0',
          theme: 'light',
          darkTheme: 'transparent_dark',
          themeURL: '',
          lang: 'en',
        },
      }
    : null,
  search: {
    provider: 'kbar',
    kbarConfig: {
      searchDocumentsPath: `${process.env.BASE_PATH || ''}/search.json`,
    },
  },
}

module.exports = siteMetadata
