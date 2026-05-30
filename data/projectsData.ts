interface Project {
  title: string
  description: string
  href?: string
  imgSrc?: string
}

const projectsData: Project[] = [
  {
    title: 'Handloom Design Generation',
    description: `Generating traditional handloom designs using Conditional GANs and Image-to-Image Translation. Engineering final year project combining deep learning with textile design.`,
    href: 'https://github.com/rajatkb/Handloom-Design-Generation-using-Deep-Neural-Networks',
    imgSrc: 'https://i.imgur.com/aU7noHS.gif',
  },
  {
    title: 'Residual Dense Network for Super-Resolution',
    description: `Keras implementation of Residual Dense Network (RDN) for single image super-resolution. Reproduces the paper with training pipelines and pretrained weights.`,
    href: 'https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras',
    imgSrc: 'https://i.imgur.com/N8rGCsf.png',
  },
  {
    title: 'DBPN — Deep Back-Projection Network',
    description: `Keras implementation of Deep Back-Projection Network for super-resolution. An alternative approach using iterative up-and-down projection stages.`,
    href: 'https://github.com/rajatkb/DBPN-Deep_Back_Projection_Network-Keras',
    imgSrc: 'https://i.imgur.com/CxDOxAQ.png',
  },
  {
    title: 'redo-paper',
    description: `Reproducing research papers from scratch. Code and model weights for various papers — mostly code since I don't have tons of H100s with me.`,
    href: 'https://github.com/rajatkb/redo-paper',
  },
  {
    title: 'Go Promise',
    description: `Promise/Future concurrency library for Go, inspired by Bluebird. Provides async primitives with composable promise chains for goroutine coordination.`,
    href: 'https://github.com/rajatkb/go-promise',
  },
  {
    title: 'Go Skip List',
    description: `Skip list data structure implementation in Golang. A probabilistic alternative to balanced trees with O(log n) search, insert, and delete operations.`,
    href: 'https://github.com/rajatkb/go-skiplist',
  },
]

export default projectsData
