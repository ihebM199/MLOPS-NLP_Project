/**
 * Example gallery for the /analyze page.
 *
 * The original repo pointed at /examples/*.jpg produced by
 * `scripts/export_examples.py` (needs the 140k dataset locally). This build
 * maps the same 8-tile gallery (4 real / 4 fake) onto the sample images the
 * Django backend already serves from `media/samples/`, so the tiles work out
 * of the box.
 */
export const EXAMPLES = [
  { id: 'real-1', label: 'Réelle', src: '/media/samples/sample_real_1.jpg' },
  { id: 'real-2', label: 'Réelle', src: '/media/samples/sample_real_2.jpg' },
  { id: 'real-3', label: 'Réelle', src: '/media/samples/sample_real_3.jpg' },
  { id: 'real-4', label: 'Réelle', src: '/media/samples/sample_real_4.jpg' },
  { id: 'fake-1', label: 'Générée', src: '/media/samples/sample_fake_1.jpg' },
  { id: 'fake-2', label: 'Générée', src: '/media/samples/sample_fake_2.jpg' },
  { id: 'fake-3', label: 'Générée', src: '/media/samples/sample_fake_3.jpg' },
  { id: 'fake-4', label: 'Générée', src: '/media/samples/sample_fake_4.jpg' },
]
