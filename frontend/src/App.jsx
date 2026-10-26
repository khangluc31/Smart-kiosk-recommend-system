import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, AudioLines, Check, ChefHat, CircleHelp, ClipboardList, Mic, Minus, Plus, ShoppingBag, Sparkles, TrendingUp, UserRound, X } from 'lucide-react';

// ---------------------------------------------------------------------------
// Cấu hình
//   VITE_API_URL : địa chỉ backend FastAPI. Bỏ trống khi chạy local (vite proxy /api -> :8000)
// ---------------------------------------------------------------------------
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '') + '/api';

const QUESTIONS = [
  { title: 'Hôm nay anh/chị muốn dùng món gì ạ?', hint: 'Ví dụ: Mình thích gà rán giòn và Pepsi.', key: 'want', placeholder: 'Tôi muốn ăn...' },
  { title: 'Anh/chị có lưu ý gì về dị ứng hoặc thành phần nào không muốn dùng hôm nay không ạ?', hint: 'Nếu không có, anh/chị có thể chọn “Không có”.', key: 'avoid', placeholder: 'Tôi không muốn ăn...' },
  { title: 'Dạ hôm nay nhà mình/nhóm mình đi bao nhiêu người ạ?', hint: 'Để KBuddy gợi ý khẩu phần phù hợp.', key: 'party', placeholder: 'Ví dụ: 2 người' },
  { title: 'Anh/chị dự định chi tiêu trong tầm giá khoảng bao nhiêu cho bữa ăn hôm nay ạ?', hint: 'KBuddy sẽ cân nhắc ngân sách khi tối ưu giỏ hàng.', key: 'budget', placeholder: 'Ví dụ: 200.000 đồng' },
];
const EMPTY_ANSWERS = { want: '', avoid: '', party: '', budget: '' };
const WELCOME_TEXT = 'Chào mừng bạn đến với KFC! KBuddy xin chào. Hôm nay chúng mình có thể giúp gì cho bạn?';
const UPSELL_TEXT = 'Dựa trên giỏ hàng của bạn, chúng tớ gợi ý các món sau. Bạn có muốn dùng thử không ạ?';

const money = n => `${Number(n || 0).toLocaleString('vi-VN')}đ`;
const total = cart => cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
const sumQty = cart => cart.reduce((sum, item) => sum + item.quantity, 0);
const toLines = cart => cart.map(x => ({ id: x.id, quantity: x.quantity }));   // dạng backend nhận
const mergeOne = (items, meal) => items.some(x => x.id === meal.id)
  ? items.map(x => x.id === meal.id ? { ...x, quantity: x.quantity + 1 } : x)
  : [...items, { ...meal, quantity: 1 }];

async function api(path, body) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error('Không kết nối được máy chủ. Kiểm tra backend (uvicorn api:app) đã chạy chưa.');
  }
  if (!res.ok) {
    let message = 'Máy chủ gặp lỗi';
    try { message = (await res.json()).detail || message; } catch { /* empty */ }
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return res.json();
}


let currentAudio = null; 
function speak(text) {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
  }
  const url = API_BASE + '/tts?text=' + encodeURIComponent(text);
  currentAudio = new Audio(url);
  currentAudio.play().catch(e => console.log("Lỗi phát âm thanh:", e));
}

// Bàn phím ảo tiếng Việt Telex; giữ nguyên hàm speak() dùng Edge-TTS ở trên.
const VN_BASE = 'aăâeêioôơuưy';
const TONE_ROWS = [
  ['a','á','à','ả','ã','ạ'],['ă','ắ','ằ','ẳ','ẵ','ặ'],['â','ấ','ầ','ẩ','ẫ','ậ'],
  ['e','é','è','ẻ','ẽ','ẹ'],['ê','ế','ề','ể','ễ','ệ'],['i','í','ì','ỉ','ĩ','ị'],
  ['o','ó','ò','ỏ','õ','ọ'],['ô','ố','ồ','ổ','ỗ','ộ'],['ơ','ớ','ờ','ở','ỡ','ợ'],
  ['u','ú','ù','ủ','ũ','ụ'],['ư','ứ','ừ','ử','ữ','ự'],['y','ý','ỳ','ỷ','ỹ','ỵ']
];
const TONES = {s:1,f:2,r:3,x:4,j:5};
const MODIFIERS = {aa:'â',aw:'ă',ee:'ê',oo:'ô',ow:'ơ',uw:'ư',dd:'đ'};
const WORD_END = /([A-Za-zÀ-ỹĐđ]+)$/u;
function vowelInfo(c) {
  const lower = c.toLowerCase();
  for (const row of TONE_ROWS) {
    const tone = row.indexOf(lower);
    if (tone !== -1) return {base:row[0], tone};
  }
  return null;
}
function setVowel(c, base, tone) {
  const row = TONE_ROWS.find(r => r[0] === base);
  if (!row) return c;
  const v = row[tone] || row[0];
  return c === c.toUpperCase() ? v.toUpperCase() : v;
}
function telexKey(text, key) {
  if (key === '⌫') return Array.from(text).slice(0,-1).join('');
  if (key === 'Space') return text + ' ';
  if (key === 'Enter') return text + '\n';
  const match = text.match(WORD_END);
  if (!match || !/^[a-z]$/i.test(key)) return text + key;
  let word = match[1];
  const lowerKey = key.toLowerCase();
  // aa, aw, ee, oo, ow, uw, dd: biến đổi nguyên âm / phụ âm Telex.
  if ('aewoud'.includes(lowerKey)) {
    for (let i = word.length-1; i >= 0; i--) {
      const original = word[i];
      const v = vowelInfo(original);
      const base = v ? v.base : original.toLowerCase();
      const modified = MODIFIERS[base + lowerKey];
      if (modified && (v || base === 'd')) {
        const replacement = v ? setVowel(original, modified, v.tone) :
          (original === original.toUpperCase() ? modified.toUpperCase() : modified);
        return text.slice(0,-word.length) + word.slice(0,i) + replacement + word.slice(i+1);
      }
    }
  }
  // Dấu Telex: s sắc, f huyền, r hỏi, x ngã, j nặng.
  if (TONES[lowerKey]) {
    const nuclei = [];
    for (let i=0;i<word.length;i++) if (vowelInfo(word[i])) nuclei.push(i);
    if (nuclei.length) {
      // Ưu tiên nguyên âm ê/ơ/ô/â/ă/ư, còn lại thường chọn âm cuối của cụm nguyên âm.
      let target = [...nuclei].reverse().find(i => 'êơôâăư'.includes(vowelInfo(word[i]).base));
      if (target === undefined) target = nuclei[nuclei.length-1];
      const info = vowelInfo(word[target]);
      const updated = setVowel(word[target],info.base,TONES[lowerKey]);
      return text.slice(0,-word.length) + word.slice(0,target) + updated + word.slice(target+1);
    }
  }
  return text + key;
}


function VirtualKeyboard({
  value,
  onChange,
  onClose,
  mode = 'text',
  title = 'Nhập câu trả lời'
}) {
  const [shift, setShift] = useState(false);

  // Telex luôn bật đối với bàn phím chữ
  const rows = mode === 'phone'
    ? [
        ['1', '2', '3'],
        ['4', '5', '6'],
        ['7', '8', '9'],
        ['+', '0', '⌫']
      ]
    : [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm']
      ];

  const press = (key) => {
    if (key === 'Shift') {
      setShift(prev => !prev);
      return;
    }

    if (mode === 'phone' && value.length >= 12 && key !== '⌫') {
      return;
    }

    const typed = shift && key.length === 1
      ? key.toUpperCase()
      : key;

    let nextValue;

    if (typed === '⌫') {
      nextValue = Array.from(value).slice(0, -1).join('');
    } else if (typed === 'Space') {
      nextValue = value + ' ';
    } else if (mode === 'text') {
      nextValue = telexKey(value, typed);
    } else {
      nextValue = value + typed;
    }

    onChange(nextValue);

    if (shift) {
      setShift(false);
    }
  };

  return (
    <div
      className="keyboard-backdrop"
      onMouseDown={e => e.preventDefault()}
      onClick={onClose}
    >
      <div
        className="virtual-keyboard"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={e => e.stopPropagation()}
      >
        <div className="keyboard-heading">
          <strong>{title}</strong>

          <button
            type="button"
            className="keyboard-close"
            onClick={onClose}
          >
            Xong <X size={18} />
          </button>
        </div>

        <div className="keyboard-preview">
          {value || 'Chạm phím để nhập...'}
        </div>

        <div className={
          mode === 'phone'
            ? 'keyboard-numpad'
            : 'keyboard-qwerty'
        }>
          {rows.map((row, i) => (
            <div className="keyboard-row" key={i}>
              {row.map(key => (
                <button
                  type="button"
                  key={key}
                  onClick={() => press(key)}
                >
                  {shift ? key.toUpperCase() : key}
                </button>
              ))}
            </div>
          ))}

          {mode === 'text' && (
            <div className="keyboard-row keyboard-controls">
              <button
                type="button"
                onClick={() => press('Shift')}
              >
                ⇧ Shift
              </button>

              <button
                type="button"
                onClick={() => press(',')}
              >
                ,
              </button>

              <button
                type="button"
                className="keyboard-space"
                onClick={() => press('Space')}
              >
                Dấu cách
              </button>

              <button
                type="button"
                onClick={() => press('.')}
              >
                .
              </button>

              <button
                type="button"
                onClick={() => press('⌫')}
              >
                ⌫
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


function Brand({ compact = false }) {
  return <div className={`brand ${compact ? 'brand--compact' : ''}`}>
    <div className="brand-mark">KFC<span className="brand-mark-bars"><i/><i/><i/></span></div>
    <div className="brand-separator"/>
    <div className="brand-buddy">k<span>buddy</span><small>SMART ORDERING</small></div>
  </div>;
}
function Btn({ children, onClick, secondary = false, outline = false, disabled = false, icon, className = '' }) {
  return <button disabled={disabled} onClick={onClick} className={`btn ${secondary ? 'btn-secondary' : ''} ${outline ? 'btn-outline' : ''} ${className}`}>{children}{icon && <ArrowRight size={18}/>}</button>;
}
// Ảnh thật từ menu KFC; không có ảnh hoặc tải lỗi -> emoji như cũ
function MealImage({ meal, small = false }) {
  const [failed, setFailed] = useState(false);
  if (meal.image && !failed) {
    return <img className={small ? 'meal-thumb' : 'meal-img'} src={meal.image} alt={meal.name} loading="lazy"
      referrerPolicy="no-referrer" onError={() => setFailed(true)} />;
  }
  return small ? <>{meal.emoji || '🍗'}</> : <><span>{meal.emoji || '🍗'}</span><span className="food-pattern">KFC</span></>;
}
function FoodCard({ meal, onAdd, compact = false }) {
  return <article className={`food-card ${compact ? 'food-card--small' : ''}`}>
    <div className="food-art"><MealImage meal={meal}/>{meal.is_hot && <span className="food-badge">HOT</span>}</div>
    <div className="food-details"><div className="food-name">{meal.name}</div><div className="food-description">{meal.description || 'Lựa chọn dành cho bạn'}</div>
      <div className="food-bottom"><b>{money(meal.price)}</b><button className="add-btn" onClick={() => onAdd(meal)} aria-label={`Thêm ${meal.name}`}><Plus size={19}/></button></div>
    </div>
  </article>;
}
function CartLines({ cart, onChange }) {
  return cart.map(item => <div className="cart-row" key={item.id}>
    <div className="cart-row-emoji"><MealImage meal={item} small/></div><div className="cart-row-main"><strong>{item.name}</strong><span>{money(item.price)}</span><div className="qty"><button onClick={() => onChange(item.id, -1)} aria-label="Giảm"><Minus size={14}/></button><b>{item.quantity}</b><button onClick={() => onChange(item.id, 1)} aria-label="Tăng"><Plus size={14}/></button></div></div>
  </div>);
}
function MiniCart({ cart, onChange, onSubmit, busy, budget }) {
  const over = budget && total(cart) > budget;
  return <aside className="cart-panel"><div className="cart-panel-title"><div><span className="eyebrow">ĐƠN HÀNG CỦA BẠN</span><h3>Giỏ hàng <span className="cart-count">{sumQty(cart)}</span></h3></div><ShoppingBag size={24}/></div>
    <div className="cart-items">{cart.length ? <CartLines cart={cart} onChange={onChange}/> : <div className="empty-cart"><ShoppingBag size={31}/><p>Chưa có món nào trong giỏ hàng.</p><small>Thêm món bạn yêu thích để tiếp tục nhé!</small></div>}</div>
    <div className="cart-footer"><div className="cart-total"><span>Tạm tính</span><strong>{money(total(cart))}</strong></div>
      {budget ? <div className={`budget-line ${over ? 'budget-line--over' : ''}`}>Ngân sách {money(budget)}{over ? ' · đang vượt' : ''}</div> : null}
      <Btn onClick={onSubmit} disabled={!cart.length || busy} icon>{busy ? 'Đang tối ưu...' : 'Tiếp tục tối ưu giỏ hàng'}</Btn></div>
  </aside>;
}

export default function App() {
  const [step, setStep] = useState('welcome');
  const [choice, setChoice] = useState('suggest');
  const [fullName, setFullName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [keyboardTarget, setKeyboardTarget] = useState(null);
  const [answers, setAnswers] = useState(EMPTY_ANSWERS);
  const [qIndex, setQIndex] = useState(0);
  const [session, setSession] = useState(null);        // {session_id, budget, route, ...} từ /session
  const [meals, setMeals] = useState([]);
  const [category, setCategory] = useState('Tất cả');
  const [recs, setRecs] = useState([]);
  const [cart, setCart] = useState([]);
  const [optimized, setOptimized] = useState(null);    // kết quả ILP1 từ /optimize-cart
  const [upgrade, setUpgrade] = useState(null);        // ILP2: lên combo
  const [upsells, setUpsells] = useState([]);          // ILP3: món mua thêm
  const [upsellInfo, setUpsellInfo] = useState({});    // {budget, left}
  const [order, setOrder] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [recording, setRecording] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => () => { if (currentAudio) currentAudio.pause(); recognitionRef.current?.abort(); }, []);
  useEffect(() => { if (step === 'welcome') speak(WELCOME_TEXT); }, []);
  useEffect(() => { if (step === 'questions') speak(QUESTIONS[qIndex].title); }, [step, qIndex]);
  useEffect(() => {
    if (step === 'optimize') speak(optimized?.changed ? 'Đây là thực đơn đã được chúng tôi tối ưu hóa cho bạn. Anh chị có đồng ý không ạ?' : 'Giỏ hàng của bạn đã là lựa chọn tiết kiệm nhất.');
    if (step === 'upsell') speak(UPSELL_TEXT);
  }, [step]);

  function go(next) { setError(''); setKeyboardTarget(null); setStep(next); }
  function add(item) { setCart(prev => mergeOne(prev, item)); }
  function change(id, diff) { setCart(prev => prev.map(x => x.id === id ? { ...x, quantity: x.quantity + diff } : x).filter(x => x.quantity > 0)); }
  function reset() {
    recognitionRef.current?.abort(); setRecording(false);
    setCart([]); setAnswers(EMPTY_ANSWERS); setQIndex(0); setFullName(''); setPhoneNumber('');
    setSession(null); setRecs([]); setOptimized(null); setUpgrade(null); setUpsells([]); setOrder(null);
    setCategory('Tất cả'); go('welcome');
  }
  async function run(fn) {
    setBusy(true); setError('');
    try { await fn(); } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  // Bước 1: tạo phiên + lấy top 5 gợi ý (two tower / LinUCB + ranker)
  function startRecommendations(allAnswers, asGuest) {
    return run(async () => {
      const login = asGuest ? {} : { name: fullName.trim(), phone: phoneNumber.trim() };
      const [menu, res] = await Promise.all([api('/meals'), api('/session', { ...login, answers: allAnswers })]);
      setMeals(menu.meals || []); setRecs(res.meals || []); setSession(res); go('menu');
    });
  }
  // Bước 2: ILP1 — giỏ rẻ nhất có đúng những món khách chọn
  function optimizeCart() {
    return run(async () => {
      const res = await api('/optimize-cart', { session_id: session.session_id, cart: toLines(cart) });
      setOptimized(res); go('optimize');
    });
  }
  // Bước 3: ILP2 (lên combo) + Item2vec + ILP3 (mua thêm) trên giỏ đã chốt
  function goUpsell(useOptimized) {
    const finalCart = useOptimized ? optimized.cart : cart;
    setCart(finalCart);
    return run(async () => {
      const res = await api('/upsell', { session_id: session.session_id, cart: toLines(finalCart) });
      setUpgrade(res.upgrade); setUpsells(res.meals || []); setUpsellInfo({ budget: res.budget, left: res.left }); go('upsell');
    });
  }
  function applyUpgrade() { setCart(upgrade.cart); setUpgrade(null); speak('Đã nâng cấp lên combo cho bạn.'); }
  // Bước 4: lưu đơn vào database (LinUCB học từ đơn này)
  function finish() {
    return run(async () => {
      const res = await api('/checkout', { session_id: session.session_id, cart: toLines(cart) });
      setOrder(res); go('done'); speak('Cảm ơn bạn đã lựa chọn KFC. Chúc bạn ngon miệng!');
    });
  }

  // Nhận giọng nói bằng Web Speech API của trình duyệt (Chrome / Edge)
  function recordVoice() {
    if (recording) { recognitionRef.current?.stop(); return; }
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) { setError('Trình duyệt không hỗ trợ nhận giọng nói. Hãy dùng Chrome/Edge hoặc nhập bằng chữ.'); return; }
    const key = QUESTIONS[qIndex].key;
    const rec = new Recognition();
    rec.lang = 'vi-VN'; rec.interimResults = true; rec.continuous = false;
    rec.onresult = (event) => {
      const text = Array.from(event.results).map(r => r[0].transcript).join(' ');
      setAnswers(prev => ({ ...prev, [key]: text }));
    };
    rec.onerror = (event) => setError(event.error === 'not-allowed' ? 'Không có quyền dùng micro. Vui lòng nhập câu trả lời bằng chữ.' : 'Không nhận diện được giọng nói, bạn thử lại nhé.');
    rec.onend = () => setRecording(false);
    if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; }
    recognitionRef.current = rec; rec.start(); setRecording(true); setError('');
  }

  const currentQ = QUESTIONS[qIndex];
  const phoneOk = /^(0|\+84)[0-9]{9}$/.test(phoneNumber.trim());
  const categories = ['Tất cả', ...new Set(meals.map(m => m.category).filter(Boolean))];
  const shownMeals = category === 'Tất cả' ? meals : meals.filter(m => m.category === category);
  const afterLogin = (asGuest) => {
    setKeyboardTarget(null); setQIndex(0);
    if (choice === 'planned') startRecommendations(EMPTY_ANSWERS, asGuest);
    else { setSession(asGuest ? { guest: true } : null); go('questions'); }
  };

  return <div className="app-shell">
    <div className="texture texture-top" aria-hidden="true">KFC · KBUDDY · KFC · KBUDDY</div>
    <header className="site-header"><Brand compact/><div className="header-right"><span className="header-live"><span className="live-dot"/> TRỢ LÝ ĐẶT MÓN THÔNG MINH</span><button className="header-reset" onClick={reset}>Bắt đầu lại</button></div></header>
    <div className="page-content">

    {step === 'welcome' && <div className="welcome screen-enter"><div className="welcome-left"><span className="pill"><Sparkles size={15}/> XIN CHÀO, MỪNG BẠN ĐẾN VỚI KFC</span><h1>Một bữa ăn ngon.<br/><em>Theo cách của bạn.</em></h1><p>Chào mừng bạn đến với KFC! Mình là <strong>KBuddy</strong>, người bạn đồng hành giúp bạn chọn món nhanh và phù hợp hơn.</p><div className="welcome-buttons"><Btn onClick={() => go('choice')} icon>Bắt đầu đặt món</Btn></div><div className="welcome-hint"><span>01. TRÒ CHUYỆN</span><span>02. CHỌN MÓN</span><span>03. THƯỞNG THỨC</span></div></div></div>}

    {step === 'choice' && <div className="center-layout screen-enter"><div className="section-heading"><span className="eyebrow">CHÀO BẠN ĐẾN VỚI KBUDDY</span><h1>Hôm nay bạn muốn<br/><em>bắt đầu thế nào?</em></h1><p>Chọn cách đặt món phù hợp nhất với bạn nhé.</p></div><div className="choice-grid"><button className="choice-card choice-card--active" onClick={() => { setChoice('suggest'); go('login'); }}><span className="choice-icon"><Sparkles size={28}/></span><strong>Tôi chưa biết ăn gì hôm nay</strong><p>Để KBuddy lắng nghe và gợi ý món ăn phù hợp với sở thích của bạn.</p><span className="choice-link">Khám phá cùng KBuddy <ArrowRight size={19}/></span></button><button className="choice-card" onClick={() => { setChoice('planned'); go('login'); }}><span className="choice-icon"><ClipboardList size={28}/></span><strong>Tôi đã có kế hoạch</strong><p>Biết mình muốn ăn gì rồi? Khám phá thực đơn và chọn trực tiếp.</p><span className="choice-link">Xem thực đơn <ArrowRight size={19}/></span></button></div><button className="text-back" onClick={() => go('welcome')}><ArrowLeft size={17}/> Quay lại</button></div>}

    {step === 'login' && <div className="center-layout screen-enter">
      <div className="section-heading"><span className="eyebrow">CÁ NHÂN HÓA TRẢI NGHIỆM</span><h1>Rất vui được<br /><em>gặp bạn!</em></h1><p>Vui lòng nhập họ tên và số điện thoại, hoặc tiếp tục với tư cách khách.</p></div>
      <div className="login-box">
        <span className="login-symbol"><UserRound size={29} /></span>
        <label htmlFor="customer-name">Họ và tên</label>
        <input id="customer-name" type="text" inputMode="none" placeholder="Nguyễn Văn A" value={fullName} onChange={e => setFullName(e.target.value)} onFocus={() => setKeyboardTarget('name')} onClick={() => setKeyboardTarget('name')} />
        <label htmlFor="customer-phone">Số điện thoại</label>
        <input id="customer-phone" type="tel" inputMode="none" placeholder="Chạm để nhập số điện thoại" value={phoneNumber}
          onChange={e => setPhoneNumber(e.target.value.replace(/[^\d+]/g, ''))}
          onFocus={() => setKeyboardTarget('phone')} onClick={() => setKeyboardTarget('phone')} />
        {phoneNumber && !phoneOk && <p className="field-error">Số điện thoại gồm 10 số, bắt đầu bằng 0 hoặc +84.</p>}
        <Btn icon disabled={!fullName.trim() || !phoneOk || busy} onClick={() => afterLogin(false)}>{busy ? 'Đang tải...' : 'Tiếp tục'}</Btn>
        <div className="login-divider">HOẶC</div>
        <button className="guest-btn" disabled={busy} onClick={() => afterLogin(true)}>Tiếp tục không đăng nhập <ArrowRight size={17} /></button>
        <p className="login-note">Họ tên + số điện thoại giúp KBuddy nhận ra bạn lần sau và gợi ý theo các món bạn hay gọi.</p>
      </div>
      <button className="text-back" onClick={() => go('choice')}><ArrowLeft size={17} /> Quay lại</button>
    </div>}

    {step === 'questions' && <div className="questions-layout screen-enter"><div className="question-main"><div className="question-top"><span className="eyebrow">KBUDDY ĐANG LẮNG NGHE</span><span className="question-number">CÂU HỎI {String(qIndex + 1).padStart(2, '0')} / 04</span></div><div className="progress-track"><div style={{ width: `${(qIndex + 1) * 25}%` }}/></div><div className="question-symbol"><AudioLines size={34}/></div><h1>{currentQ.title}</h1><p className="question-hint">{currentQ.hint}</p>
      <div className="answer-area"><label htmlFor="answer">CÂU TRẢ LỜI CỦA BẠN</label>
        <textarea id="answer" inputMode="none" placeholder={currentQ.placeholder} value={answers[currentQ.key]} onChange={e => setAnswers(prev => ({ ...prev, [currentQ.key]: e.target.value }))} onFocus={() => setKeyboardTarget('answer')} onClick={() => setKeyboardTarget('answer')} />
        <div className="answer-actions">
          <button className={`mic-btn ${recording ? 'mic-btn--on' : ''}`} onClick={recordVoice} disabled={busy}><Mic size={19}/>{recording ? 'Đang nghe... chạm để dừng' : 'Trả lời bằng giọng nói'}</button>
          {currentQ.key === 'avoid' && <button className="skip-small" onClick={() => setAnswers(prev => ({ ...prev, avoid: 'Không có' }))}>Không có</button>}
        </div></div>
      <div className="question-nav"><button className="text-back" onClick={() => qIndex ? setQIndex(qIndex - 1) : go('login')}><ArrowLeft size={17}/> Quay lại</button><Btn disabled={busy} icon onClick={() => qIndex < 3 ? setQIndex(qIndex + 1) : startRecommendations(answers, session?.guest)}>{busy ? 'Đang tìm món...' : qIndex < 3 ? 'Tiếp theo' : 'Xem món gợi ý'}</Btn></div></div>
      <aside className="question-aside"><span className="aside-label">MỘT CHÚT VỀ BẠN</span><h3>Món ngon được<br/>chọn riêng cho bạn.</h3><p>KBuddy sẽ dựa vào sở thích, nhu cầu và ngân sách để đưa ra gợi ý.</p><div className="question-checks">{QUESTIONS.map((q, i) => <div key={q.key} className={`question-check ${i === qIndex ? 'question-check--active' : ''}`}><span>{i < qIndex ? <Check size={15}/> : `0${i + 1}`}</span><div>{['Sở thích','Lưu ý ăn uống','Số người','Ngân sách'][i]}</div></div>)}</div><div className="aside-motto">YOUR TASTE.<br/>YOUR CHOICE.</div></aside></div>}

    {step === 'menu' && <div className="menu-layout screen-enter"><div className="menu-content">
      <div className="page-title"><div><span className="eyebrow">GỢI Ý DÀNH RIÊNG CHO BẠN</span><h1>Hôm nay ăn gì <em>ngon?</em></h1><p>{session?.is_regular ? 'Dựa trên những lần bạn đã gọi món, KBuddy nghĩ bạn sẽ thích:' : 'Đây là những lựa chọn KBuddy nghĩ bạn sẽ thích.'}</p></div><span className="tag-yellow"><Sparkles size={15}/> AI RECOMMENDATION</span></div>
      {(session?.budget || session?.unwant?.length > 0) && <div className="notice">
        {session.budget ? <span>Ngân sách: <b>{money(session.budget)}</b></span> : null}
        {session.unwant?.length > 0 && <span>Không dùng: <b>{session.unwant.join(', ')}</b></span>}
      </div>}
      <div className="thin-line"/>
      <div className="block-title"><h2>Được đề xuất cho bạn</h2><span>TOP {recs.length} MÓN</span></div>
      <div className="food-grid">{recs.map(m => <FoodCard key={m.id} meal={m} onAdd={add}/>)}</div>
      <div className="block-title block-title--second"><h2>Khám phá thêm món ngon</h2><span>THỰC ĐƠN KFC</span></div>
      <div className="chips">{categories.map(c => <button key={c} className={`chip ${c === category ? 'chip--active' : ''}`} onClick={() => setCategory(c)}>{c}</button>)}</div>
      <div className="food-grid">{shownMeals.map(m => <FoodCard key={m.id} meal={m} onAdd={add}/>)}</div>
    </div><MiniCart cart={cart} onChange={change} onSubmit={optimizeCart} busy={busy} budget={session?.budget}/></div>}

    {step === 'optimize' && optimized && <div className="review-layout screen-enter">
      <div className="section-heading"><span className="eyebrow">KBUDDY ĐÃ HOÀN THÀNH</span>
        {optimized.changed
          ? <><h1>Thực đơn đã được<br/><em>tối ưu cho bạn.</em></h1><p>Cùng những món bạn chọn, gộp thành combo sẽ tiết kiệm <b>{money(optimized.saving)}</b>. Bạn có muốn áp dụng không?</p></>
          : <><h1>Giỏ hàng của bạn<br/><em>đã tối ưu rồi.</em></h1><p>Không có cách gộp combo nào rẻ hơn cho những món bạn chọn.</p></>}
      </div>
      <div className="review-box"><div className="review-top"><div><span className="eyebrow">GIỎ HÀNG ĐỀ XUẤT</span><h3>{optimized.changed ? 'Gợi ý tối ưu' : 'Giỏ hàng của bạn'}</h3></div><ChefHat size={28}/></div>
        {optimized.cart.map(item => <div className="review-row" key={item.id}><span className="review-emoji"><MealImage meal={item} small/></span><strong>{item.name} <small>× {item.quantity}</small></strong><b>{money(item.price * item.quantity)}</b></div>)}
        <div className="review-total"><span>Tổng đề xuất</span><strong>{money(optimized.cost)}</strong></div>
        <div className="review-compare">Giỏ hiện tại: {money(optimized.original_cost)} <span>·</span> Đề xuất: {money(optimized.cost)}{optimized.over_budget && <><span>·</span><em className="over">vượt ngân sách {money(optimized.budget)}</em></>}</div>
        <div className="review-buttons">{optimized.changed
          ? <><Btn secondary onClick={() => goUpsell(false)} disabled={busy}>Không, giữ giỏ cũ</Btn><Btn onClick={() => goUpsell(true)} disabled={busy} icon>{busy ? 'Đang xử lý...' : 'Có, dùng giỏ đề xuất'}</Btn></>
          : <><Btn secondary onClick={() => go('menu')} disabled={busy}>Quay lại chọn món</Btn><Btn onClick={() => goUpsell(false)} disabled={busy} icon>{busy ? 'Đang xử lý...' : 'Tiếp tục'}</Btn></>}</div>
      </div><p className="disclaimer">Bạn luôn có quyền giữ lại giỏ hàng ban đầu.</p></div>}

    {step === 'upsell' && <div className="upsell-layout screen-enter">
      <div className="section-heading"><span className="eyebrow">THÊM MỘT CHÚT NGON MIỆNG</span><h1>Món ngon hơn<br/><em>khi có bạn đồng hành.</em></h1><p>Dựa vào giỏ hàng hiện tại, KBuddy muốn giới thiệu thêm một vài món. Bạn có muốn thử không?</p></div>
      {upgrade && <div className="upgrade-card">
        <span className="choice-icon"><TrendingUp size={26}/></span>
        <div className="upgrade-main"><span className="eyebrow">NÂNG CẤP LÊN COMBO</span>
          <h3>Thêm {money(upgrade.extra_cost)} để có {upgrade.cart.map(x => `${x.quantity > 1 ? x.quantity + '× ' : ''}${x.name}`).join(' + ')}</h3>
          {Object.keys(upgrade.gained || {}).length > 0 && <p>Bạn được thêm: {Object.entries(upgrade.gained).map(([k, n]) => `${n} ${k}`).join(', ')}</p>}
          {upgrade.over_budget && <p className="over">Lưu ý: tổng {money(upgrade.cost)} sẽ vượt ngân sách {money(upsellInfo.budget)}.</p>}</div>
        <Btn onClick={applyUpgrade} icon>Nâng cấp</Btn>
      </div>}
      <div className="upsell-cards">{upsells.length ? upsells.map(m => <FoodCard key={m.id} meal={m} onAdd={add}/>) : <p className="upsell-empty">{upsellInfo.left != null && upsellInfo.left < 20000 ? `Giỏ hàng đã gần chạm ngân sách (còn ${money(Math.max(upsellInfo.left, 0))}), KBuddy không gợi ý thêm để bạn không bị vượt.` : 'Hiện chưa có gợi ý bổ sung phù hợp với giỏ hàng này.'}</p>}</div>
      <div className="upsell-bottom"><div className="upsell-cart"><span>GIỎ HÀNG HIỆN TẠI</span><strong>{sumQty(cart)} món · {money(total(cart))}</strong><div className="cart-items cart-items--inline"><CartLines cart={cart} onChange={change}/></div></div><Btn icon onClick={finish} disabled={!cart.length || busy}>{busy ? 'Đang đặt món...' : 'Hoàn tất lựa chọn'}</Btn></div>
    </div>}

    {step === 'done' && <div className="center-layout end-screen screen-enter"><div className="done-mark"><Check size={42}/></div><span className="eyebrow">CẢM ƠN BẠN!</span><h1>Chúc bạn có một<br/><em>bữa ăn thật ngon.</em></h1><p>Đơn hàng <b>#{order?.order_id}</b> đã được ghi nhận. Vui lòng thanh toán tại quầy.</p><div className="done-receipt"><span>{sumQty(cart)} món đã chọn</span><strong>{money(order?.total ?? total(cart))}</strong></div><Btn icon onClick={reset}>Về màn hình chính</Btn></div>}

    {error && <div className="error-toast" role="alert"><CircleHelp size={18}/><span>{error}</span><button onClick={() => setError('')}><X size={17}/></button></div>}
    </div>
    {keyboardTarget && <VirtualKeyboard
      key={keyboardTarget === 'answer' ? `answer-${qIndex}` : keyboardTarget}
      mode={keyboardTarget === 'phone' ? 'phone' : 'text'}
      title={keyboardTarget === 'phone' ? 'Nhập số điện thoại' : keyboardTarget === 'name' ? 'Nhập họ và tên' : 'Nhập câu trả lời'}
      value={keyboardTarget === 'phone' ? phoneNumber : keyboardTarget === 'name' ? fullName : answers[currentQ.key]}
      onChange={keyboardTarget === 'phone' ? setPhoneNumber : keyboardTarget === 'name' ? setFullName : value => setAnswers(prev => ({ ...prev, [currentQ.key]: value }))}
      onClose={() => setKeyboardTarget(null)}
    />}
    <footer className="footer"><span>© KBUDDY · SMART KFC ORDERING</span><span>MADE TO MAKE YOUR DAY BETTER <span className="footer-star">✳</span></span></footer>
  </div>;
}