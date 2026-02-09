/**
 * @fileoverview Conversation widget.
 */

/** @param {{ model: !DOMWidgetModel, el: !HTMLElement }} context */
function render({model, el}) {
  el.innerHTML = `<div class="conversation">${model.get('conversation')}</div>`;

  addCollapseEvents(el);
  addResizeImages(el);
  connectStream(el, model);
}
export default {render};


/**
 * Add listener to the collapse button.
 * @param {!HTMLElement} conv The conversation element.
 */
function addCollapseEvents(conv) {
  const allDetails = conv.querySelectorAll('details');

  allDetails.forEach((details) => {
    toogleIcons(details);

    const summary = details.querySelector(':scope > summary');
    const button = summary.querySelector('.collapse-icon');
    if (!button) return;

    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      const icon = button.querySelector('use');
      const isCollapsed = icon.getAttribute('href') === '#icon-expand';

      if (isCollapsed && !details.open) {
        details.open = true;
      }

      // Update all icons
      const allIcons = details.querySelectorAll('.collapse-icon use');
      allIcons.forEach((curr_icon) => {
        if (isCollapsed) {
          curr_icon.setAttribute('href', '#icon-collapse');
        } else {
          curr_icon.setAttribute('href', '#icon-expand');
        }
      });

      // Update all children
      const children = details.querySelectorAll('details');
      children.forEach((child) => {
        if (isCollapsed) {
          child.open = true;
        } else {
          child.open = false;
        }
      });
    });
  });
}


/**
 * Toggles the icons in the summary.
 * @param {!HTMLElement} details The details element.
 */
function toogleIcons(details) {
  const titleIcon = details.querySelector(':scope > summary .title-icon');

  details.addEventListener('toggle', (event) => {
    // Only show the emoji icons when the details are closed.
    if (details.open) {
      titleIcon.style.visibility = 'hidden';
    } else {
      titleIcon.style.visibility = 'visible';
    }
  });
}

/**
 * Adds a click event listener to each image to resize them.
 * @param {!HTMLElement} conv The conversation element.
 */
function addResizeImages(conv) {
  // Select all <img> elements on the page
  const images = conv.querySelectorAll('img');

  // Add a click event listener to each image
  images.forEach((img) => {
    // Do not process missing images
    if (img.classList.contains('missing')) {
      return;
    }

    // Set a pointer cursor to indicate the images are clickable
    img.style.cursor = 'pointer';

    img.addEventListener('click', () => {
      // Get the actual, rendered height of the image
      const currentHeight =
          window.getComputedStyle(img).getPropertyValue('height');

      // Toggle the height
      if (currentHeight === '150px') {
        img.style.height = 'auto';  // Switch to full height
      } else {
        img.style.height = '150px';  // Switch to 150px
      }
    });
  });
}

/**
 * Connects the stream to the widget.
 * @param {!HTMLElement} el The conversation element.
 * @param {!DOMWidgetModel} model The model.
 */
function connectStream(el, model) {
  const stream = el.querySelector('.stream-tmp');
  model.on('change:stream_replace_tmp_html', () => {
    stream.innerHTML = model.get('stream_replace_tmp_html');
  });
  model.on('change:stream_add_final_html', () => {
    stream.insertAdjacentHTML(
        'beforebegin', model.get('stream_add_final_html'));
  });
}
