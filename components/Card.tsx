import Image from './Image'
import Link from './Link'
import Tag from './Tag'
import { ArrowRight } from 'lucide-react'

const Card = ({ title, description, imgSrc, href, year, tags }) => (
  <div className="md max-w-[544px] p-4 md:w-1/2">
    <div className="h-full overflow-hidden rounded-md border-2 border-gray-200/60 dark:border-gray-700/60"
    >
      {imgSrc &&
        (href ? (
          <Link href={href} aria-label={`Link to ${title}`}>
            <Image
              alt={title}
              src={imgSrc}
              className="object-cover object-center md:h-36 lg:h-48"
              width={544}
              height={306}
              unoptimized
            />
          </Link>
        ) : (
          <Image
            alt={title}
            src={imgSrc}
            className="object-cover object-center md:h-36 lg:h-48"
            width={544}
            height={306}
            unoptimized
          />
        ))}
      <div className="flex flex-col p-6">
        <div className="flex-1">
          <div className="mb-2 flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
            {year && <span>{year}</span>}
          </div>
          <h2 className="mb-3 text-2xl leading-8 font-bold tracking-tight">
            {href ? (
              <Link href={href} aria-label={`Link to ${title}`}>
                {title}
              </Link>
            ) : (
              title
            )}
          </h2>
          <p className="prose mb-3 max-w-none text-gray-500 dark:text-gray-400">{description}</p>
        </div>
        {tags && tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5 py-3">
            {tags.map((tag) => (
              <Tag key={tag} text={tag} />
            ))}
          </div>
        )}
        {href && (
          <Link
            href={href}
            className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400 text-base leading-6 font-medium"
            aria-label={`Link to ${title}`}
          >
            <span className="inline-flex items-center gap-1">
              Learn more
              <ArrowRight size={14} className="inline-block" />
            </span>
          </Link>
        )}
      </div>
    </div>
  </div>
)

export default Card
