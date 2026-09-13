/**
 * The two element constructors the pages share.
 *
 * Every page builds its content in JavaScript — there is no template engine and
 * no build step — so without these the modules fill up with four-line
 * createElement/className/textContent sequences. `card.js` had its own copy
 * before this existed; the dysbiosis page had none and was twenty-nine copies
 * of the long form.
 */

const NS = 'http://www.w3.org/2000/svg';

/**
 * Create an HTML element.
 *
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text] - Set as textContent, never as innerHTML.
 * @returns {HTMLElement}
 */
export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Create an SVG element, which needs the namespace to render at all.
 *
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {SVGElement}
 */
export function svgElement(tag, className, text) {
  const node = document.createElementNS(NS, tag);
  if (className) node.setAttribute('class', className);
  if (text !== undefined) node.textContent = text;
  return node;
}
