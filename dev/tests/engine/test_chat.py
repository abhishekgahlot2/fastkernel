import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
CHAT = Path(__file__).resolve().parents[3] / "server" / "chat.html"

# Execute the complete production script with a minimal DOM and controlled I/O.
# Tests drive the registered UI handlers; no copied production handler or network.
HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync(process.argv[1], 'utf8')
  .match(/<script>([\s\S]*?)<\/script>/)[1];

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.handlers = {};
    this.value = '';
    this.style = {};
    this.options = [];
    this.children = [];
    this.scrollHeight = 40;
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  addEventListener(name, callback) { this.handlers[name] = callback; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  remove() {} focus() {} setAttribute() {} removeAttribute() {}
  querySelector() { return new Element(); }
}

function createChat(storage = new Map(), writable = true) {
  const elements = {};
  const requests = [];
  const reads = [];
  for (const id of ['chat', 'form', 'input', 'attachments', 'image-input',
                   'attach', 'effort', 'api-key', 'send', 'recents', 'new-chat',
                   'mobile-new', 'menu', 'scrim']) elements[id] = new Element();
  elements.effort.options = ['xhigh', 'medium', 'low', 'none'].map(value => ({value}));
  class FileReader {
    readAsDataURL(file) {
      this.result = `data:image/png;base64,${file.name}`;
      reads.push(this);
    }
  }
  const context = vm.createContext({
    document: {
      querySelector: selector => elements[selector.slice(1)] || null,
      createElement: tag => new Element(tag),
      body: new Element(),
    },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem(key, value) {
        if (!writable) throw new Error('Storage is full');
        storage.set(key, value);
      },
    },
    scrollTo() {}, AbortController, TextDecoder, Uint8Array, console, FileReader,
    crypto: require('node:crypto').webcrypto,
    fetch(url, options) {
      return new Promise((resolve, reject) => {
        requests.push({url, headers: options.headers, body: JSON.parse(options.body), resolve, reject});
        options.signal.addEventListener('abort', () => {
          reject(Object.assign(new Error('Stopped'), {name: 'AbortError'}));
        });
      });
    },
  });
  vm.runInContext(script, context);
  return {elements, requests, reads};
}

const flush = () => new Promise(resolve => setImmediate(resolve));
const submit = chat => chat.elements.form.handlers.submit({preventDefault() {}});
const enter = chat => chat.elements.input.handlers.keydown({
  key: 'Enter', keyCode: 13, shiftKey: false, isComposing: false,
  preventDefault() {},
});
function setText(chat, value) {
  chat.elements.input.value = value;
  chat.elements.input.handlers.input();
}
function pasteImages(chat, ...names) {
  const start = chat.reads.length;
  chat.elements.input.handlers.paste({
    clipboardData: {
      items: names.map(name => ({type: 'image/png', getAsFile: () => ({type: 'image/png', name})})),
      getData: () => '',
    },
    preventDefault() {},
  });
  return chat.reads.slice(start);
}
const pasteImage = (chat, name) => pasteImages(chat, name)[0];
function selectImages(chat, ...names) {
  const start = chat.reads.length;
  chat.elements['image-input'].files = names.map(name => ({type: 'image/png', name}));
  chat.elements['image-input'].handlers.change();
  return chat.reads.slice(start);
}
const imagePart = name => ({type: 'image_url', image_url: {url: `data:image/png;base64,${name}`}});
const draftImages = chat => chat.elements.attachments.children.flatMap(
  item => item.children.filter(child => child.tagName === 'IMG').map(image => image.src)
);
function removeImage(chat, index) {
  const item = chat.elements.attachments.children[index];
  item.children.find(child => child.tagName === 'BUTTON').handlers.click();
}

function succeed(request) {
  let sent = false;
  request.resolve({ok: true, body: {getReader: () => ({
    async read() {
      if (sent) return {done: true};
      sent = true;
      return {done: false, value: Buffer.from(
        'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'
      )};
    },
  })}});
}
"""


@unittest.skipUnless(NODE, "Node.js is required to execute the chat UI tests")
class ChatTest(unittest.TestCase):
    def run_chat(self, program):
        result = subprocess.run(
            [NODE, "-e", HARNESS + program, str(CHAT)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_key_is_sent_only_as_a_header_and_not_persisted(self):
        self.run_chat(r"""
const storage = new Map();
const chat = createChat(storage);
chat.elements['api-key'].value = 'test-server-key';
setText(chat, 'hello');
submit(chat);
assert.equal(chat.requests[0].headers.Authorization, 'Bearer test-server-key');
assert.ok(!JSON.stringify(chat.requests[0].body).includes('test-server-key'));
assert.ok(!JSON.stringify([...storage]).includes('test-server-key'));
const fresh = createChat(storage);
assert.equal(fresh.elements['api-key'].value, '');
setText(fresh, 'hello');
submit(fresh);
assert.equal(fresh.requests[0].headers.Authorization, undefined);
""")

    def test_restores_saved_chats_and_effort_with_read_only_storage(self):
        self.run_chat(r"""
const saved = JSON.stringify([{id: 'saved', title: 'Saved chat', updated: 1,
  messages: [{role: 'user', content: 'remembered message'}]}]);
for (const writable of [true, false]) {
  const storage = new Map([
    ['splash-chats', saved], ['splash-thinking-effort', 'low'],
  ]);
  const chat = createChat(storage, writable);
  assert.equal(chat.elements.effort.value, 'low');
  assert.equal(chat.elements.recents.children.length, 1);
  chat.elements.recents.children[0].handlers.click();
  setText(chat, 'continue');
  submit(chat);
  assert.equal(chat.requests[0].body.messages[0].content, 'remembered message');
  assert.equal(storage.get('splash-thinking-effort'), 'low');
}
""")

    def test_enter_preserves_composition_and_shift_but_sends_normal_input(self):
        self.run_chat(r"""
for (const [name, overrides, shouldSend] of [
  ['composition', {isComposing: true}, false],
  ['Safari composition end', {isComposing: false, keyCode: 229}, false],
  ['Shift+Enter', {shiftKey: true}, false],
  ['other key', {key: 'a'}, false],
  ['Enter', {}, true],
]) {
  const {elements, requests} = createChat();
  elements.input.value = 'hello';
  let prevented = false;
  elements.input.handlers.keydown({
    key: 'Enter', keyCode: 13, shiftKey: false, isComposing: false,
    ...overrides,
    preventDefault() { prevented = true; },
  });
  assert.equal(requests.length, Number(shouldSend), name);
  assert.equal(prevented, shouldSend, name);
  assert.equal(elements.input.value, shouldSend ? '' : 'hello', name);
  if (shouldSend) {
    assert.equal(requests[0].url, '/v1/chat/completions');
    assert.deepEqual(requests[0].body.messages, [{role: 'user', content: 'hello'}]);
  }
}
""")

    def test_completion_preserves_the_next_draft_and_restores_failed_attachments(self):
        self.run_chat(r"""
(async () => {
  for (const outcome of ['success', 'stop', 'error']) {
    for (const draft of ['empty', 'ready image', 'loading image']) {
      const chat = createChat();
      const {elements, requests} = chat;
      pasteImage(chat, 'original').onload();
      await flush();
      elements.input.value = 'first prompt';
      submit(chat);
      const original = [{type: 'text', text: 'first prompt'}, imagePart('original')];
      assert.deepEqual(requests[0].body.messages, [{role: 'user', content: original}]);
      assert.deepEqual(draftImages(chat), []);
      let reading;
      if (draft !== 'empty') {
        elements.input.value = 'next prompt';
        reading = pasteImage(chat, 'next');
        if (draft === 'ready image') {
          reading.onload();
          await flush();
          assert.deepEqual(draftImages(chat), [imagePart('next').image_url.url]);
        }
      }
      assert.equal(elements.send.disabled, false, 'Stop stays available while an image loads');
      if (outcome === 'success') succeed(requests[0]);
      else if (outcome === 'stop') submit(chat);
      else requests[0].resolve({ok: false, json: async () => ({error: {message: 'Failed'}})});
      await flush();
      assert.equal(elements.attach.disabled, false, `${outcome}: request must finish`);
      if (draft === 'loading image') {
        assert.equal(elements.send.disabled, true, `${outcome}: unfinished next image blocks Send`);
        submit(chat);
        enter(chat);
        assert.equal(requests.length, 1, `${outcome}: cannot submit an unfinished next image`);
        reading.onload();
        await flush();
        assert.equal(requests.length, 1, 'finishing a read must not queue a request');
      }
      const restored = outcome === 'success' ? [] : [imagePart('original')];
      const expectedImages = [...restored, ...(draft === 'empty' ? [] : [imagePart('next')])];
      const expectedText = draft !== 'empty' ? 'next prompt' : outcome === 'success' ? '' : 'first prompt';
      assert.equal(elements.input.value, expectedText, `${outcome}, ${draft}`);
      assert.deepEqual(draftImages(chat), expectedImages.map(part => part.image_url.url), `${outcome}, ${draft}`);
      // The prior serialized request must stay unchanged as the next draft grows.
      assert.deepEqual(requests[0].body.messages, [{role: 'user', content: original}]);
      if (expectedText || expectedImages.length) {
        submit(chat);
        const expected = [...(expectedText ? [{type: 'text', text: expectedText}] : []), ...expectedImages];
        assert.deepEqual(requests[1].body.messages.at(-1), {role: 'user', content: expected});
        // Successful history remains; stopped/failed requests are removed.
        assert.equal(requests[1].body.messages.length, outcome === 'success' ? 3 : 1);
        succeed(requests[1]);
        await flush();
      }
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_loading_image_blocks_send_and_enter_until_ready(self):
        self.run_chat(r"""
(async () => {
  for (const text of ['describe this image', '']) {
    const chat = createChat();
    const {elements, requests} = chat;
    setText(chat, text);
    const reading = pasteImage(chat, 'before-submit');
    // Dispatching submit directly also exercises the guard behind the disabled button.
    submit(chat);
    enter(chat);
    assert.equal(requests.length, 0, 'neither submit nor Enter may send partial content');
    assert.equal(elements.attachments.children.length, 1, 'selection is visible immediately');
    assert.deepEqual(draftImages(chat), [], 'unfinished image is a placeholder');
    assert.equal(elements.send.disabled, true, 'Send is disabled while reading');
    assert.equal(elements.input.value, text);
    reading.onload();
    await flush();
    assert.equal(requests.length, 0, 'completion does not automatically submit');
    assert.equal(elements.send.disabled, false, 'image-only requests become sendable too');
    assert.deepEqual(draftImages(chat), [imagePart('before-submit').image_url.url]);
    if (text) submit(chat);
    else enter(chat);
    assert.deepEqual(requests[0].body.messages, [{role: 'user', content: [
      ...(text ? [{type: 'text', text}] : []), imagePart('before-submit'),
    ]}]);
    succeed(requests[0]);
    await flush();
    assert.deepEqual(draftImages(chat), []);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_separate_selections_keep_order_when_reads_finish_out_of_order(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  const [first, second] = selectImages(chat, 'first', 'second');
  const [third] = selectImages(chat, 'third');
  assert.equal(chat.elements.attachments.children.length, 3);
  third.onload();
  second.onload();
  await flush();
  assert.deepEqual(draftImages(chat), ['second', 'third'].map(name => imagePart(name).image_url.url));
  assert.equal(chat.elements.send.disabled, true, 'one pending image blocks the whole request');
  submit(chat);
  assert.equal(chat.requests.length, 0);
  first.onload();
  await flush();
  assert.deepEqual(draftImages(chat), ['first', 'second', 'third'].map(name => imagePart(name).image_url.url));
  submit(chat);
  assert.deepEqual(chat.requests[0].body.messages, [{role: 'user', content:
    ['first', 'second', 'third'].map(imagePart),
  }]);
  succeed(chat.requests[0]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_failed_image_remains_removable_and_preserves_other_selected_images(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  setText(chat, 'first prompt');
  submit(chat);
  const [failed, ready] = pasteImages(chat, 'failed', 'ready');
  failed.error = new Error('Image could not be read');
  failed.onerror();
  ready.onload();
  await flush();
  assert.equal(chat.elements.send.disabled, false, 'Stop remains available with a failed next image');
  submit(chat);
  await flush();
  assert.equal(chat.elements.attach.disabled, false, 'request was stopped');
  const failedItem = chat.elements.attachments.children[0];
  assert.equal(chat.elements.attachments.children.length, 2, 'failure remains visible');
  assert.ok(failedItem.children.some(child => child.tagName === 'SPAN' && child.textContent));
  assert.deepEqual(draftImages(chat), [imagePart('ready').image_url.url]);
  setText(chat, 'keep the readable image');
  assert.equal(chat.elements.send.disabled, true, 'failed image must be removed before sending');
  submit(chat);
  enter(chat);
  assert.equal(chat.requests.length, 1);
  removeImage(chat, 0);
  assert.equal(chat.elements.attachments.children.length, 1);
  assert.equal(chat.elements.send.disabled, false);
  submit(chat);
  assert.deepEqual(chat.requests[1].body.messages, [{role: 'user', content: [
    {type: 'text', text: 'keep the readable image'}, imagePart('ready'),
  ]}]);
  succeed(chat.requests[1]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_removing_pending_image_prevents_late_result_from_returning(self):
        self.run_chat(r"""
(async () => {
  for (const outcome of ['load', 'error']) {
    const chat = createChat();
    const reading = pasteImage(chat, 'removed');
    removeImage(chat, 0);
    assert.equal(chat.elements.attachments.children.length, 0);
    assert.equal(chat.elements.send.disabled, true, 'empty composer stays disabled');
    const next = pasteImage(chat, 'next');
    if (outcome === 'load') reading.onload();
    else {
      reading.error = new Error('Late read failure');
      reading.onerror();
    }
    await flush();
    assert.equal(chat.elements.attachments.children.length, 1, 'removed image cannot return');
    assert.deepEqual(draftImages(chat), []);
    assert.equal(chat.elements.send.disabled, true, 'next image is still loading');
    next.onload();
    await flush();
    submit(chat);
    assert.deepEqual(chat.requests[0].body.messages, [{role: 'user', content: [imagePart('next')]}]);
    succeed(chat.requests[0]);
    await flush();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_switching_chats_ignores_results_from_previous_draft(self):
        self.run_chat(r"""
(async () => {
  for (const navigation of ['new chat', 'recent chat']) {
    for (const outcome of ['load', 'error']) {
      const chat = createChat();
      setText(chat, 'saved conversation');
      submit(chat);
      succeed(chat.requests[0]);
      await flush();
      const reading = pasteImage(chat, 'old-chat');
      if (navigation === 'new chat') chat.elements['new-chat'].handlers.click();
      else chat.elements.recents.children[0].handlers.click();
      assert.equal(chat.elements.attachments.children.length, 0);
      const next = pasteImage(chat, 'new-chat');
      if (outcome === 'load') reading.onload();
      else {
        reading.error = new Error('Old chat read failure');
        reading.onerror();
      }
      await flush();
      assert.equal(chat.elements.attachments.children.length, 1);
      assert.deepEqual(draftImages(chat), []);
      assert.equal(chat.elements.send.disabled, true);
      next.onload();
      await flush();
      enter(chat);
      assert.deepEqual(chat.requests[1].body.messages.at(-1), {role: 'user', content: [imagePart('new-chat')]});
      assert.equal(chat.requests[1].body.messages.length, navigation === 'new chat' ? 1 : 3);
      succeed(chat.requests[1]);
      await flush();
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


if __name__ == "__main__":
    unittest.main()
