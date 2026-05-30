import projectsData from '@/data/projectsData'
import Card from '@/components/Card'
import { genPageMetadata } from 'app/seo'

export const metadata = genPageMetadata({ title: 'Projects' })

export default function Projects() {
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
            ML systems, super-resolution, databases, and distributed tooling — things I've built and
            shipped.
          </p>
        </div>
      </section>

      <div className="container py-12">
        <div className="-m-4 flex flex-wrap">
          {projectsData.map((d) => (
            <Card
              key={d.title}
              title={d.title}
              description={d.description}
              imgSrc={d.imgSrc}
              href={d.href}
            />
          ))}
        </div>
      </div>
    </>
  )
}
