/**
 * @fileoverview Conversation string widget.
 */

/** @param {{ model: !DOMWidgetModel, el: !HTMLElement }} context */
function render({model, el}) {
  el.innerHTML = model.get('conversation');

  // addCollapseEvents(el);
}
export default {render};


/**
 * Add listener to the collapse button.
 * @param {!HTMLElement} conv The conversation element.
 */
function addCollapseEvents(conv) {
  const allTags = conv.querySelectorAll('.tag');

  allTags.forEach((tag) => {
    const content = tag.querySelector('.content');

    tag.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      // TODO(epot): Supports double click, using timeout ? setTimeout
      setTimeout(() => {
        // Ignore clicks when selecting text.
        const selection = window.getSelection();
        if (selection.toString().length > 0) {
          return;
        }

        content.classList.toggle('collapsed');
      }, 250);
    });
  });
}
