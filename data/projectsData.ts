interface Project {
  title: string
  description: string
  href?: string
  imgSrc?: string
  year: number
  tags: string[]
}

const projectsData: Project[] = [
  {
    title: 'Boro DB',
    description: `A simple database written in Go. Minimalist storage engine with basic query capabilities — built to understand database internals from the ground up.`,
    href: 'https://github.com/rajatkb/boro-db',
    year: 2025,
    tags: ['go', 'database', 'systems'],
  },
  {
    title: 'Go Skip List',
    description: `Skip list data structure implementation in Golang. A probabilistic alternative to balanced trees with O(log n) search, insert, and delete operations.`,
    href: 'https://github.com/rajatkb/go-skiplist',
    year: 2022,
    tags: ['go', 'algorithms', 'data-structures'],
  },
  {
    title: 'Go Promise',
    description: `Promise/Future concurrency library for Go, inspired by Bluebird. Provides async primitives with composable promise chains for goroutine coordination.`,
    href: 'https://github.com/rajatkb/go-promise',
    year: 2020,
    tags: ['go', 'concurrency', 'async'],
  },
  {
    title: 'Handloom Design Generation',
    description: `Generating traditional handloom designs using Conditional GANs and Image-to-Image Translation. Engineering final year project combining deep learning with textile design.`,
    href: 'https://github.com/rajatkb/Handloom-Design-Generation-using-Deep-Neural-Networks',
    imgSrc: 'https://i.imgur.com/aU7noHS.gif',
    year: 2018,
    tags: ['deep-learning', 'computer-vision', 'generative-models'],
  },
  {
    title: 'DBPN — Deep Back-Projection Network',
    description: `Keras implementation of Deep Back-Projection Network for super-resolution. An alternative approach using iterative up-and-down projection stages.`,
    href: 'https://github.com/rajatkb/DBPN-Deep_Back_Projection_Network-Keras',
    imgSrc: 'https://i.imgur.com/CxDOxAQ.png',
    year: 2018,
    tags: ['deep-learning', 'super-resolution', 'keras'],
  },
  {
    title: 'Residual Dense Network for Super-Resolution',
    description: `Keras implementation of Residual Dense Network (RDN) for single image super-resolution. Reproduces the paper with training pipelines and pretrained weights.`,
    href: 'https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras',
    imgSrc: 'https://i.imgur.com/N8rGCsf.png',
    year: 2018,
    tags: ['deep-learning', 'super-resolution', 'keras'],
  },
]

export default projectsData
