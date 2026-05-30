interface Project {
  title: string
  description: string
  href?: string
  imgSrc?: string
}

const projectsData: Project[] = [
  {
    title: 'Paper Shelf',
    description: `A curated collection of ML research paper notes — transformer architectures, multi-token prediction, and more.`,
    href: '/blog',
  },
  {
    title: 'Interactive Demos',
    description: `Live, interactive code demos exploring ML concepts, algorithms, and system designs. Built with MDX + custom React components.`,
    href: '/blog',
  },
]

export default projectsData
