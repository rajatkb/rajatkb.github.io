import Link from '@/components/Link'
import Tag from '@/components/Tag'
import siteMetadata from '@/data/siteMetadata'
import projectsData from '@/data/projectsData'
import { formatDate } from 'pliny/utils/formatDate'

const MAX_DISPLAY = 5

const HERO_IMAGES = [
  'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1200&q=80',
  'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&q=80',
  'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1200&q=80',
]

export default function Home({ posts }) {
  const heroImage = HERO_IMAGES[0]

  // Merge projects and posts, sort newest first
  const projects = projectsData.map((p) => ({
    type: 'project',
    title: p.title,
    description: p.description,
    href: p.href,
    date: `${p.year}-01-01`,
    year: p.year,
    tags: p.tags,
    slug: p.href,
  }))

  const blogPosts = posts.map((p) => ({
    type: 'post',
    title: p.title,
    description: p.summary,
    href: `/blog/${p.slug}`,
    date: p.date,
    year: new Date(p.date).getFullYear(),
    tags: p.tags,
    slug: p.slug,
  }))

  const all = [...projects, ...blogPosts].sort((a, b) => {
    const da = new Date(a.date).getTime()
    const db = new Date(b.date).getTime()
    return db - da
  })

  return (
    <>
      {/* Hero Section */}
      <section className="relative mb-12 overflow-hidden rounded-2xl">
        <div className="absolute inset-0">
          <img src={heroImage} alt="" className="h-full w-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-r from-black/70 to-black/30" />
        </div>
        <div className="relative z-10 px-8 py-20 sm:px-12 sm:py-28">
          <h1 className="font-serif text-4xl font-bold tracking-tight text-white sm:text-5xl md:text-6xl">
            {siteMetadata.title}
          </h1>
          <p className="mt-4 max-w-2xl text-lg leading-relaxed text-gray-200 sm:text-xl">
            {siteMetadata.description}
          </p>
          <div className="mt-6 flex flex-wrap gap-4">
            <Link
              href="/about"
              className="rounded-lg bg-white/20 px-6 py-2 text-sm font-semibold text-white backdrop-blur-sm transition hover:bg-white/30"
            >
              About Me
            </Link>
            <Link
              href="/projects"
              className="rounded-lg bg-white/20 px-6 py-2 text-sm font-semibold text-white backdrop-blur-sm transition hover:bg-white/30"
            >
              View Projects
            </Link>
          </div>
        </div>
      </section>

      {/* Combined Feed */}
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="space-y-2 pt-6 pb-8 md:space-y-5">
          <h1 className="text-3xl leading-9 font-extrabold tracking-tight text-gray-900 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14 dark:text-gray-100">
            Latest
          </h1>
          <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
            Projects and writings
          </p>
        </div>
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {all.slice(0, MAX_DISPLAY).map((item) => (
            <li key={item.slug} className="py-12">
              <article>
                <div className="space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0">
                  <dl>
                    <dt className="sr-only">Date</dt>
                    <dd className="text-base leading-6 font-medium text-gray-500 dark:text-gray-400">
                      <span>{item.year}</span>
                      <span className="ml-2 text-xs uppercase tracking-wider text-primary-500">
                        {item.type === 'project' ? 'Project' : 'Post'}
                      </span>
                    </dd>
                  </dl>
                  <div className="space-y-5 xl:col-span-3">
                    <div className="space-y-6">
                      <div>
                        <h2 className="text-2xl leading-8 font-bold tracking-tight">
                          <Link
                            href={item.href}
                            className="text-gray-900 dark:text-gray-100"
                          >
                            {item.title}
                          </Link>
                        </h2>
                        <div className="flex flex-wrap">
                          {item.tags?.map((tag) => (
                            <Tag key={tag} text={tag} />
                          ))}
                        </div>
                      </div>
                      <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                        {item.description}
                      </div>
                    </div>
                    <div className="text-base leading-6 font-medium">
                      <Link
                        href={item.href}
                        className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
                      >
                        {item.type === 'project' ? 'View project &rarr;' : 'Read more &rarr;'}
                      </Link>
                    </div>
                  </div>
                </div>
              </article>
            </li>
          ))}
        </ul>
      </div>
      <div className="flex justify-end gap-4 text-base leading-6 font-medium">
        <Link
          href="/projects"
          className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
        >
          All Projects &rarr;
        </Link>
        <Link
          href="/blog"
          className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
        >
          All Posts &rarr;
        </Link>
      </div>
    </>
  )
}
