from lxml import etree
import re

# HTML内容
# 请将您的HTML文件内容粘贴到此处
html_content = """
[<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>商品评论 - 无线蓝牙耳机</title>
    <!-- 引入Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- 引入Font Awesome -->
    <link href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css" rel="stylesheet">
    
    <!-- 配置Tailwind自定义颜色和字体 -->
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#3B82F6',
                        secondary: '#F59E0B',
                        neutral: {
                            100: '#F3F4F6',
                            200: '#E5E7EB',
                            300: '#D1D5DB',
                            400: '#9CA3AF',
                            500: '#6B7280',
                            600: '#4B5563',
                            700: '#374151',
                            800: '#1F2937',
                            900: '#111827',
                        }
                    },
                    fontFamily: {
                        inter: ['Inter', 'system-ui', 'sans-serif'],
                    },
                }
            }
        }
    </script>
    
    <!-- 自定义工具类 -->
    <style type="text/tailwindcss">
        @layer utilities {
            .content-auto {
                content-visibility: auto;
            }
            .text-shadow {
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card-hover {
                transition: all 0.3s ease;
            }
            .card-hover:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            }
        }
    </style>
</head>
<body class="font-inter bg-neutral-100 text-neutral-800 min-h-screen">
    <!-- 导航栏 -->
    <header class="sticky top-0 z-50 bg-white shadow-md transition-all duration-300">
        <div class="container mx-auto px-4 py-3 flex justify-between items-center">
            <a href="#" class="text-primary font-bold text-xl flex items-center space-x-2">
                <i class="fa fa-headphones"></i>
                <span>SoundWave</span>
            </a>
            
            <nav class="hidden md:flex space-x-8">
                <a href="#" class="text-neutral-600 hover:text-primary transition-colors">首页</a>
                <a href="#" class="text-neutral-600 hover:text-primary transition-colors">商品</a>
                <a href="#" class="text-primary font-medium border-b-2 border-primary pb-1">评论</a>
                <a href="#" class="text-neutral-600 hover:text-primary transition-colors">关于我们</a>
            </nav>
            
            <div class="flex items-center space-x-4">
                <button class="text-neutral-600 hover:text-primary transition-colors">
                    <i class="fa fa-search text-lg"></i>
                </button>
                <button class="text-neutral-600 hover:text-primary transition-colors">
                    <i class="fa fa-shopping-cart text-lg"></i>
                </button>
                <button class="md:hidden text-neutral-600 hover:text-primary transition-colors">
                    <i class="fa fa-bars text-xl"></i>
                </button>
            </div>
        </div>
    </header>

    <main class="container mx-auto px-4 py-8">
        <!-- 商品信息和总体评分 -->
        <section class="mb-12 bg-white rounded-xl shadow-lg overflow-hidden">
            <div class="md:flex">
                <!-- 商品图片 -->
                <div class="md:w-1/3 p-6 flex items-center justify-center bg-neutral-100">
                    <img src="https://picsum.photos/seed/headphones/500/500" alt="无线蓝牙耳机" class="rounded-lg shadow-md max-h-64 object-contain">
                </div>
                
                <!-- 商品信息 -->
                <div class="md:w-2/3 p-6 md:p-8">
                    <div class="flex flex-col md:flex-row md:items-center justify-between mb-4">
                        <h1 class="text-2xl md:text-3xl font-bold text-neutral-900 mb-2 md:mb-0">SoundWave Pro 无线蓝牙耳机</h1>
                        <div class="text-2xl font-bold text-primary">¥399</div>
                    </div>
                    
                    <p class="text-neutral-600 mb-6">高保真音质，主动降噪，IPX7防水，长达30小时续航，舒适佩戴体验</p>
                    
                    <!-- 总体评分 -->
                    <div class="mb-6">
                        <div class="flex items-center mb-2">
                            <div class="text-3xl font-bold text-neutral-900 mr-3">4.7</div>
                            <div class="text-secondary">
                                <i class="fa fa-star"></i>
                                <i class="fa fa-star"></i>
                                <i class="fa fa-star"></i>
                                <i class="fa fa-star"></i>
                                <i class="fa fa-star-half-o"></i>
                            </div>
                            <div class="ml-3 text-neutral-500">(236 条评价)</div>
                        </div>
                        
                        <!-- 评分分布 -->
                        <div class="space-y-2">
                            <div class="flex items-center">
                                <span class="w-10 text-sm">5 星</span>
                                <div class="flex-1 mx-3 h-2 bg-neutral-200 rounded-full overflow-hidden">
                                    <div class="h-full bg-secondary rounded-full" style="width: 75%"></div>
                                </div>
                                <span class="w-10 text-sm text-right">75%</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-10 text-sm">4 星</span>
                                <div class="flex-1 mx-3 h-2 bg-neutral-200 rounded-full overflow-hidden">
                                    <div class="h-full bg-secondary rounded-full" style="width: 15%"></div>
                                </div>
                                <span class="w-10 text-sm text-right">15%</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-10 text-sm">3 星</span>
                                <div class="flex-1 mx-3 h-2 bg-neutral-200 rounded-full overflow-hidden">
                                    <div class="h-full bg-secondary rounded-full" style="width: 6%"></div>
                                </div>
                                <span class="w-10 text-sm text-right">6%</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-10 text-sm">2 星</span>
                                <div class="flex-1 mx-3 h-2 bg-neutral-200 rounded-full overflow-hidden">
                                    <div class="h-full bg-secondary rounded-full" style="width: 2%"></div>
                                </div>
                                <span class="w-10 text-sm text-right">2%</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-10 text-sm">1 星</span>
                                <div class="flex-1 mx-3 h-2 bg-neutral-200 rounded-full overflow-hidden">
                                    <div class="h-full bg-secondary rounded-full" style="width: 2%"></div>
                                </div>
                                <span class="w-10 text-sm text-right">2%</span>
                            </div>
                        </div>
                    </div>
                    
                    <button class="bg-primary hover:bg-primary/90 text-white font-medium py-2 px-6 rounded-lg transition-colors shadow-md hover:shadow-lg">
                        查看商品详情
                    </button>
                </div>
            </div>
        </section>
        
        <!-- 评论筛选和排序 -->
        <section class="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <h2 class="text-2xl font-bold text-neutral-900">用户评论</h2>
            
            <div class="flex flex-wrap gap-3">
                <!-- 筛选器 -->
                <div class="relative">
                    <select class="appearance-none bg-white border border-neutral-300 rounded-lg py-2 pl-4 pr-10 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary">
                        <option value="all">所有评分</option>
                        <option value="5">5 星</option>
                        <option value="4">4 星及以上</option>
                        <option value="3">3 星及以上</option>
                        <option value="2">2 星及以上</option>
                        <option value="1">1 星</option>
                    </select>
                    <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-neutral-500">
                        <i class="fa fa-chevron-down text-xs"></i>
                    </div>
                </div>
                
                <!-- 排序 -->
                <div class="relative">
                    <select class="appearance-none bg-white border border-neutral-300 rounded-lg py-2 pl-4 pr-10 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary">
                        <option value="newest">最新</option>
                        <option value="highest">评分最高</option>
                        <option value="lowest">评分最低</option>
                        <option value="mostHelpful">最有帮助</option>
                    </select>
                    <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-neutral-500">
                        <i class="fa fa-chevron-down text-xs"></i>
                    </div>
                </div>
                
                <!-- 筛选标签 -->
                <div class="flex gap-2">
                    <button class="bg-primary/10 text-primary text-sm py-2 px-4 rounded-full hover:bg-primary/20 transition-colors">
                        音质
                    </button>
                    <button class="bg-primary/10 text-primary text-sm py-2 px-4 rounded-full hover:bg-primary/20 transition-colors">
                        续航
                    </button>
                    <button class="bg-primary/10 text-primary text-sm py-2 px-4 rounded-full hover:bg-primary/20 transition-colors">
                        降噪
                    </button>
                    <button class="bg-neutral-200 text-neutral-700 text-sm py-2 px-4 rounded-full hover:bg-neutral-300 transition-colors">
                        <i class="fa fa-plus mr-1"></i> 更多
                    </button>
                </div>
            </div>
        </section>
        
        <!-- 评论列表 -->
        <section class="space-y-6 mb-12">
            <!-- 评论1 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <!-- 用户信息 -->
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user1/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">李明</h3>
                        <p class="text-sm text-neutral-500">2023年11月15日</p>
                    </div>
                    
                    <!-- 评论内容 -->
                    <div class="md:w-5/6">
                        <!-- 评分 -->
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <!-- 标题和内容 -->
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">音质超乎想象，降噪效果出色</h4>
                        <p class="text-neutral-700 mb-4">这款耳机的音质真的让我惊艳，低音浑厚有力，高音清晰不刺耳。降噪功能也非常棒，在嘈杂的地铁里也能安静地享受音乐。电池续航也很给力，充一次电可以用差不多一周（我每天听2-3小时）。佩戴起来也很舒适，长时间使用不会觉得耳朵痛。总之，这是一款性价比很高的耳机，强烈推荐！</p>
                        
                        <!-- 评论图片 -->
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review1a/200/200" alt="耳机外观展示" class="w-24 h-24 object-cover rounded-lg">
                            <img src="https://picsum.photos/seed/review1b/200/200" alt="耳机配件展示" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <!-- 互动按钮 -->
                        <div class="flex items-center gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (42)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论2 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user2/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">王芳</h3>
                        <p class="text-sm text-neutral-500">2023年11月10日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">整体不错，但有些小缺点</h4>
                        <p class="text-neutral-700 mb-4">耳机的音质和续航都很满意，连接也很稳定。但是有两个小问题：一是耳机在运动时容易掉，不太适合跑步；二是麦克风在嘈杂环境下收音效果一般，打电话时对方说有时听不清。如果主要是用来听音乐，这款耳机很推荐，但如果经常需要运动时使用或者频繁接电话，可能需要考虑一下。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (18)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                        
                        <!-- 商家回复 -->
                        <div class="mt-4 pt-4 border-t border-neutral-200 bg-neutral-50 p-4 rounded-lg">
                            <div class="flex items-center gap-2 mb-2">
                                <img src="https://picsum.photos/seed/shop/100/100" alt="商家头像" class="w-8 h-8 rounded-full object-cover">
                                <span class="font-medium text-primary">SoundWave官方旗舰店</span>
                                <span class="text-xs text-neutral-500">2023年11月11日</span>
                            </div>
                            <p class="text-neutral-700 text-sm">感谢您的反馈！关于佩戴问题，我们有配套的耳翼配件可以增强稳定性，已为您补发。麦克风方面我们会在后续固件更新中优化，感谢您的支持！</p>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论3 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user3/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">张伟</h3>
                        <p class="text-sm text-neutral-500">2023年11月5日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">性价比之王，超出预期</h4>
                        <p class="text-neutral-700 mb-4">这个价位能买到这样的耳机真的很值！我之前用的是某知名品牌的耳机，价格是这个的两倍多，但说实话，音质和续航感觉差别不大。这款耳机的降噪虽然比不上顶级旗舰，但日常使用完全足够。APP的功能也很丰富，可以自定义音效，调整降噪强度。客服响应也很及时，有问题都能很快解决。已经推荐给身边的朋友了。</p>
                        
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review3a/200/200" alt="耳机与手机配对界面" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (56)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论4 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user4/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">刘静</h3>
                        <p class="text-sm text-neutral-500">2023年10月28日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-half-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">还不错，但有提升空间</h4>
                        <p class="text-neutral-700 mb-4">耳机外观设计很时尚，我很喜欢。音质方面中高音表现不错，但低音有点不足，希望后续固件更新能改善。续航能力确实很强，我一周充一次电就够了。佩戴舒适度一般，长时间戴会有点不舒服。降噪在低频噪音（如空调声）方面效果很好，但对人声的降噪效果一般。总体来说，399元的价格还是很合理的。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (12)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论5 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user5/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">赵强</h3>
                        <p class="text-sm text-neutral-500">2023年10月20日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">完美适配我的工作需求</h4>
                        <p class="text-neutral-700 mb-4">作为一个经常需要开视频会议的上班族，这款耳机简直是救星。降噪功能让我在嘈杂的家里也能清晰地听到同事讲话，麦克风的收音效果也很好，对方说我的声音很清晰。电池续航特别给力，一周充一次电完全够用。最惊喜的是多点连接功能，可以同时连接电脑和手机，电话来了自动切换，非常方便。强烈推荐给需要经常远程办公的朋友！</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (37)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论6 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user6/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">陈雪</h3>
                        <p class="text-sm text-neutral-500">2023年10月15日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                            <i class="fa fa-star-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">不太满意，连接经常断开</h4>
                        <p class="text-neutral-700 mb-4">可能是我运气不好，收到的这款耳机连接很不稳定，尤其是在人多的地方，经常会断开连接。音质也一般，比不上我之前用的同价位产品。客服让我重置耳机，但问题依旧。后来申请了换货，希望新的能好一点。续航确实不错，这是唯一的亮点。如果对连接稳定性要求高的话，不推荐这款。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (8)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                        
                        <!-- 商家回复 -->
                        <div class="mt-4 pt-4 border-t border-neutral-200 bg-neutral-50 p-4 rounded-lg">
                            <div class="flex items-center gap-2 mb-2">
                                <img src="https://picsum.photos/seed/shop/100/100" alt="商家头像" class="w-8 h-8 rounded-full object-cover">
                                <span class="font-medium text-primary">SoundWave官方旗舰店</span>
                                <span class="text-xs text-neutral-500">2023年10月16日</span>
                            </div>
                            <p class="text-neutral-700 text-sm">非常抱歉给您带来不好的体验！我们已安排发出新的耳机，并会对您退回的产品进行检测。如果您对新耳机仍有任何问题，请随时联系我们，我们会负责到底。</p>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论7 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user7/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">孙洋</h3>
                        <p class="text-sm text-neutral-500">2023年10月10日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">音质很好，防水性能出色</h4>
                        <p class="text-neutral-700 mb-4">买这款耳机主要是为了运动时使用，IPX7的防水等级果然没让人失望，即使在大雨中跑步也没问题。音质方面在运动耳机中算是佼佼者了，重低音很适合运动时听。佩戴也比较稳固，一般的跑步不会掉，但做剧烈运动时还是会有点松动。续航方面，每次充电可以用5-6小时，充电盒还能再充3-4次，完全够用了。</p>
                        
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review7a/200/200" alt="运动中使用耳机" class="w-24 h-24 object-cover rounded-lg">
                            <img src="https://picsum.photos/seed/review7b/200/200" alt="耳机防水测试" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (29)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论8 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user8/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">周敏</h3>
                        <p class="text-sm text-neutral-500">2023年10月5日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">颜值与实力并存</h4>
                        <p class="text-neutral-700 mb-4">首先要说的是这款耳机的颜值真的很高，我买的是白色款，简约时尚，戴出去很多朋友都问我是什么牌子的。音质方面也很出色，特别是听流行音乐和人声，非常清晰自然。降噪功能在办公室使用很合适，能有效隔绝键盘声和同事的交谈声。触摸控制也很灵敏，操作方便。总之，各方面都很满意，性价比很高。</p>
                        
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review8a/200/200" alt="白色耳机外观" class="w-24 h-24 object-cover rounded-lg">
                            <img src="https://picsum.photos/seed/review8b/200/200" alt="耳机充电盒" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (45)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论9 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user9/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">吴杰</h3>
                        <p class="text-sm text-neutral-500">2023年9月30日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-half-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">中规中矩，没有特别惊喜</h4>
                        <p class="text-neutral-700 mb-4">这款耳机整体表现中规中矩，没有特别突出的亮点，也没有明显的缺点。音质、续航、降噪都处于同价位产品的平均水平。佩戴舒适度还不错，但长时间佩戴还是会有些不适。连接稳定性很好，从没出现过断连的情况。如果追求稳定可靠，这款耳机是个不错的选择，但如果你想体验一些高端功能，可能需要考虑更贵的型号。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (15)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论10 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user10/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">郑琳</h3>
                        <p class="text-sm text-neutral-500">2023年9月25日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">对听力敏感者很友好</h4>
                        <p class="text-neutral-700 mb-4">我对音质比较敏感，很多耳机的高频会让我觉得刺耳，但这款耳机的调音很温和，长时间听也不会觉得不舒服。音量控制很精细，即使调到很低也能清晰听到，这点对保护听力很重要。APP里还有听力保护模式，会自动限制最大音量，这点非常贴心。降噪功能也很自然，不会有压迫感。总之，非常适合像我这样对听力比较敏感的人。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (22)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论11 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user11/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">林浩</h3>
                        <p class="text-sm text-neutral-500">2023年9月20日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">续航不错，但音质一般</h4>
                        <p class="text-neutral-700 mb-4">这款耳机的续航确实很出色，充一次电可以用很久。但音质方面不太满意，尤其是低音部分，感觉有点沉闷，缺乏层次感。降噪效果也一般，在嘈杂环境下效果不明显。另外，耳机的体积有点大，戴在耳朵上显得不太美观。如果对音质要求不高，只是用来听个响，这款耳机还是可以考虑的。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (9)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论12 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user12/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">黄婷</h3>
                        <p class="text-sm text-neutral-500">2023年9月15日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-half-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">很适合学生党，性价比高</h4>
                        <p class="text-neutral-700 mb-4">作为一名学生，预算有限但又想要一款各方面都不错的耳机，这款SoundWave Pro完全满足了我的需求。价格适中，音质比我之前用的几十块钱的耳机好太多了。续航也很给力，一周充一次电就够了。我经常在图书馆用，降噪功能能帮我隔绝周围的噪音，让我更专注于学习。唯一的小缺点是耳机盒有点大，不太容易放进小口袋里。</p>
                        
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review12a/200/200" alt="图书馆使用场景" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (31)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论13 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user13/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">徐杰</h3>
                        <p class="text-sm text-neutral-500">2023年9月10日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">音质调校很专业，细节丰富</h4>
                        <p class="text-neutral-700 mb-4">作为一个音乐发烧友，我对音质的要求比较高。这款耳机虽然价格不高，但音质表现非常专业。三频均衡，细节丰富，解析力不错，能听到很多以前没注意到的音乐细节。声场表现也超出预期，有一定的空间感。APP里的自定义音效功能很强大，可以根据不同类型的音乐进行调整。降噪功能在安静环境下可以关闭，开启环境音模式，既能听到周围声音又不影响音乐体验。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (52)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论14 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user14/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">郭丽</h3>
                        <p class="text-sm text-neutral-500">2023年9月5日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">电池衰减有点快</h4>
                        <p class="text-neutral-700 mb-4">买了三个月，感觉电池衰减有点明显。刚开始充满电可以用8-9小时，现在只能用5-6小时了。联系客服，说是正常现象，但感觉衰减速度有点快。音质和降噪都还不错，就是这个电池问题有点让人担心。希望能用久一点吧。另外，耳机的做工有点一般，边角有轻微的毛刺。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (14)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                        
                        <!-- 商家回复 -->
                        <div class="mt-4 pt-4 border-t border-neutral-200 bg-neutral-50 p-4 rounded-lg">
                            <div class="flex items-center gap-2 mb-2">
                                <img src="https://picsum.photos/seed/shop/100/100" alt="商家头像" class="w-8 h-8 rounded-full object-cover">
                                <span class="font-medium text-primary">SoundWave官方旗舰店</span>
                                <span class="text-xs text-neutral-500">2023年9月6日</span>
                            </div>
                            <p class="text-neutral-700 text-sm">您好，锂电池在前几个月有轻微容量衰减是正常现象，通常会稳定在80%以上。我们的产品提供18个月质保，如果您的耳机续航低于标称值的70%，可以联系我们进行检测和更换。关于做工问题，我们会反馈给生产部门加强品控，感谢您的反馈！</p>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论15 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user15/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">马明</h3>
                        <p class="text-sm text-neutral-500">2023年8月30日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">很适合旅行使用</h4>
                        <p class="text-neutral-700 mb-4">上个月带着这款耳机去旅行，表现非常出色。在飞机上，降噪功能能有效隔绝发动机的噪音，让长途飞行不再那么难熬。续航也很给力，往返航班加上中间几天的使用，只充了一次电。耳机盒很小巧，不占地方，放在随身包里很方便。音质方面，在嘈杂环境下表现依然稳定。唯一的小缺点是耳机在耳朵里戴久了会有点不舒服，但相比其他耳机已经好很多了。</p>
                        
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review15a/200/200" alt="飞机上使用耳机" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (36)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论16 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user16/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">朱燕</h3>
                        <p class="text-sm text-neutral-500">2023年8月25日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">客服服务非常好</h4>
                        <p class="text-neutral-700 mb-4">耳机本身质量很好，音质和续航都很满意。但更想表扬一下客服团队，我收到耳机后不知道怎么连接电脑，联系客服后，他们非常耐心地指导我操作，还发了详细的图文教程。后来有一次不小心把耳机盒弄丢了，客服也很快帮我安排了单独购买配件，价格也很合理。这样的服务真的让人很放心，以后还会继续支持这个品牌。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (27)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论17 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user17/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">胡军</h3>
                        <p class="text-sm text-neutral-500">2023年8月20日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                            <i class="fa fa-star-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">使用一个月后出现故障</h4>
                        <p class="text-neutral-700 mb-4">刚开始用着还不错，但一个月后右耳耳机突然没声音了。联系客服后，他们让我尝试重置，但问题依旧。后来安排了换货，新的耳机目前使用正常。虽然售后还算及时，但刚用一个月就出现故障还是让人不太满意。希望新换的这款能耐用一点。音质和续航方面确实不错，这是值得肯定的。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (11)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论18 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user18/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">何倩</h3>
                        <p class="text-sm text-neutral-500">2023年8月15日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">很适合语言学习</h4>
                        <p class="text-neutral-700 mb-4">我买这款耳机主要是用来学习英语的，它的音质清晰，人声表现特别好，听英语听力和 Podcast 非常合适。APP里有一个专注模式，可以增强人声，减弱背景音乐，对学习外语很有帮助。续航也很长，充一次电可以用好几天。耳机很轻，戴久了也不会不舒服。唯一的小缺点是触摸控制有时候会误触，希望后续固件更新能优化一下。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (23)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论19 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user19/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">罗伟</h3>
                        <p class="text-sm text-neutral-500">2023年8月10日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">比预期的好很多</h4>
                        <p class="text-neutral-700 mb-4">本来没抱太大期望，毕竟价格不贵，但收到后真的很惊喜。音质比我之前用的某大牌千元耳机差不多，甚至在某些方面还要更好。降噪功能也很实用，在办公室用可以有效隔绝外界干扰。续航更是超出预期，充一次电用一周完全没问题。佩戴舒适度也很好，经常戴着戴着就忘了摘下来。总之，这是我用过的性价比最高的耳机了。</p>
                        
                        <div class="flex gap-2 mb-4">
                            <img src="https://picsum.photos/seed/review19a/200/200" alt="耳机与笔记本电脑" class="w-24 h-24 object-cover rounded-lg">
                            <img src="https://picsum.photos/seed/review19b/200/200" alt="耳机细节图" class="w-24 h-24 object-cover rounded-lg">
                        </div>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (48)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
            
            <!-- 评论20 -->
            <article class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex flex-col md:flex-row md:items-start gap-6">
                    <div class="md:w-1/6 flex flex-col items-center md:items-start">
                        <img src="https://picsum.photos/seed/user20/100/100" alt="用户头像" class="w-14 h-14 rounded-full object-cover mb-2">
                        <h3 class="font-medium text-neutral-900">唐琳</h3>
                        <p class="text-sm text-neutral-500">2023年8月5日</p>
                    </div>
                    
                    <div class="md:w-5/6">
                        <div class="text-secondary mb-3">
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star"></i>
                            <i class="fa fa-star-half-o"></i>
                            <i class="fa fa-star-o"></i>
                        </div>
                        
                        <h4 class="text-lg font-semibold text-neutral-900 mb-2">总体不错，细节有待改进</h4>
                        <p class="text-neutral-700 mb-4">这款耳机整体表现不错，音质和续航都让人满意。但有些细节方面还有提升空间：比如充电盒的开合手感有点松，没有那种清脆的段落感；耳机的指示灯太亮了，晚上使用有点晃眼；APP的界面设计可以更简洁一些，现在有些功能找起来不太方便。希望厂商能在后续产品中改进这些小问题，那就更完美了。</p>
                        
                        <div class="flex gap-6 text-neutral-500">
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-thumbs-up"></i>
                                <span>有用 (17)</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-comment-o"></i>
                                <span>回复</span>
                            </button>
                            <button class="flex items-center gap-1 hover:text-primary transition-colors">
                                <i class="fa fa-flag-o"></i>
                                <span>举报</span>
                            </button>
                        </div>
                    </div>
                </div>
            </article>
        </section>
        
        <!-- 分页 -->
        <section class="flex justify-center mt-12">
            <nav class="inline-flex rounded-md shadow">
                <a href="#" class="relative inline-flex items-center px-2 py-2 rounded-l-md border border-neutral-300 bg-white text-sm font-medium text-neutral-500 hover:bg-neutral-50">
                    <i class="fa fa-chevron-left text-xs"></i>
                </a>
                <a href="#" class="relative inline-flex items-center px-4 py-2 border border-neutral-300 bg-primary text-sm font-medium text-white">
                    1
                </a>
                <a href="#" class="relative inline-flex items-center px-4 py-2 border border-neutral-300 bg-white text-sm font-medium text-neutral-700 hover:bg-neutral-50">
                    2
                </a>
                <a href="#" class="relative inline-flex items-center px-4 py-2 border border-neutral-300 bg-white text-sm font-medium text-neutral-700 hover:bg-neutral-50">
                    3
                </a>
                <span class="relative inline-flex items-center px-4 py-2 border border-neutral-300 bg-white text-sm font-medium text-neutral-700">
                    ...
                </span>
                <a href="#" class="relative inline-flex items-center px-4 py-2 border border-neutral-300 bg-white text-sm font-medium text-neutral-700 hover:bg-neutral-50">
                    12
                </a>
                <a href="#" class="relative inline-flex items-center px-2 py-2 rounded-r-md border border-neutral-300 bg-white text-sm font-medium text-neutral-500 hover:bg-neutral-50">
                    <i class="fa fa-chevron-right text-xs"></i>
                </a>
            </nav>
        </section>
        
        <!-- 发表评论 -->
        <section class="mt-16 bg-white rounded-xl shadow-md p-6 md:p-8">
            <h2 class="text-2xl font-bold text-neutral-900 mb-6">发表您的评论</h2>
            
            <form class="space-y-6">
                <!-- 评分 -->
                <div>
                    <label class="block text-neutral-700 font-medium mb-2">您的评分</label>
                    <div class="text-2xl text-neutral-300 hover:text-secondary transition-colors cursor-pointer" id="rating">
                        <i class="fa fa-star"></i>
                        <i class="fa fa-star"></i>
                        <i class="fa fa-star"></i>
                        <i class="fa fa-star"></i>
                        <i class="fa fa-star"></i>
                    </div>
                </div>
                
                <!-- 评论标题 -->
                <div>
                    <label for="review-title" class="block text-neutral-700 font-medium mb-2">评论标题</label>
                    <input type="text" id="review-title" class="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary" placeholder="请输入评论标题">
                </div>
                
                <!-- 评论内容 -->
                <div>
                    <label for="review-content" class="block text-neutral-700 font-medium mb-2">评论内容</label>
                    <textarea id="review-content" rows="5" class="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary" placeholder="请分享您使用产品的体验..."></textarea>
                </div>
                
                <!-- 上传图片 -->
                <div>
                    <label class="block text-neutral-700 font-medium mb-2">上传图片（可选）</label>
                    <div class="flex items-center justify-center w-full">
                        <label class="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-neutral-300 rounded-lg cursor-pointer bg-neutral-50 hover:bg-neutral-100 transition-colors">
                            <div class="flex flex-col items-center justify-center pt-5 pb-6">
                                <i class="fa fa-cloud-upload text-2xl text-neutral-400 mb-2"></i>
                                <p class="mb-1 text-sm text-neutral-600"><span class="font-semibold">点击上传</span> 或拖放文件</p>
                                <p class="text-xs text-neutral-500">支持 JPG, PNG 格式，最大 5MB</p>
                            </div>
                            <input type="file" class="hidden" multiple accept="image/*" />
                        </label>
                    </div>
                </div>
                
                <!-- 提交按钮 -->
                <div>
                    <button type="submit" class="bg-primary hover:bg-primary/90 text-white font-medium py-3 px-8 rounded-lg transition-colors shadow-md hover:shadow-lg">
                        发表评论
                    </button>
                </div>
            </form>
        </section>
    </main>
    
    <!-- 页脚 -->
    <footer class="bg-neutral-900 text-white mt-20">
        <div class="container mx-auto px-4 py-12">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
                <div>
                    <h3 class="text-xl font-bold mb-4 flex items-center">
                        <i class="fa fa-headphones mr-2"></i>
                        SoundWave
                    </h3>
                    <p class="text-neutral-400 mb-4">专注于提供高品质音频设备，让每个人都能享受纯净音质。</p>
                    <div class="flex space-x-4">
                        <a href="#" class="text-neutral-400 hover:text-white transition-colors">
                            <i class="fa fa-facebook"></i>
                        </a>
                        <a href="#" class="text-neutral-400 hover:text-white transition-colors">
                            <i class="fa fa-twitter"></i>
                        </a>
                        <a href="#" class="text-neutral-400 hover:text-white transition-colors">
                            <i class="fa fa-instagram"></i>
                        </a>
                        <a href="#" class="text-neutral-400 hover:text-white transition-colors">
                            <i class="fa fa-youtube-play"></i>
                        </a>
                    </div>
                </div>
                
                <div>
                    <h4 class="text-lg font-semibold mb-4">快速链接</h4>
                    <ul class="space-y-2">
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">首页</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">产品</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">评论</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">关于我们</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">联系我们</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="text-lg font-semibold mb-4">帮助中心</h4>
                    <ul class="space-y-2">
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">常见问题</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">保修政策</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">退换货政策</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">使用手册</a></li>
                        <li><a href="#" class="text-neutral-400 hover:text-white transition-colors">固件更新</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="text-lg font-semibold mb-4">联系我们</h4>
                    <ul class="space-y-2">
                        <li class="flex items-start">
                            <i class="fa fa-map-marker mt-1 mr-3 text-neutral-400"></i>
                            <span class="text-neutral-400">北京市朝阳区科技园A座1001室</span>
                        </li>
                        <li class="flex items-center">
                            <i class="fa fa-phone mr-3 text-neutral-400"></i>
                            <span class="text-neutral-400">400-123-4567</span>
                        </li>
                        <li class="flex items-center">
                            <i class="fa fa-envelope mr-3 text-neutral-400"></i>
                            <span class="text-neutral-400">support@soundwave.com</span>
                        </li>
                    </ul>
                </div>
            </div>
            
            <div class="border-t border-neutral-800 mt-10 pt-6 flex flex-col md:flex-row justify-between items-center">
                <p class="text-neutral-500 text-sm">© 2023 SoundWave. 保留所有权利。</p>
                <div class="flex space-x-6 mt-4 md:mt-0">
                    <a href="#" class="text-neutral-500 hover:text-white text-sm transition-colors">隐私政策</a>
                    <a href="#" class="text-neutral-500 hover:text-white text-sm transition-colors">使用条款</a>
                    <a href="#" class="text-neutral-500 hover:text-white text-sm transition-colors">Cookie 政策</a>
                </div>
            </div>
        </div>
    </footer>
    
    <!-- JavaScript -->
    <script>
        // 导航栏滚动效果
        window.addEventListener('scroll', function() {
            const header = document.querySelector('header');
            if (window.scrollY > 50) {
                header.classList.add('py-2', 'shadow-lg');
                header.classList.remove('py-3', 'shadow-md');
            } else {
                header.classList.add('py-3', 'shadow-md');
                header.classList.remove('py-2', 'shadow-lg');
            }
        });
        
        // 评分功能
        const ratingStars = document.querySelectorAll('#rating i');
        ratingStars.forEach((star, index) => {
            star.addEventListener('mouseover', () => {
                ratingStars.forEach((s, i) => {
                    if (i <= index) {
                        s.classList.add('text-secondary');
                        s.classList.remove('text-neutral-300');
                    } else {
                        s.classList.add('text-neutral-300');
                        s.classList.remove('text-secondary');
                    }
                });
            });
            
            star.addEventListener('click', () => {
                ratingStars.forEach((s, i) => {
                    if (i <= index) {
                        s.classList.add('text-secondary');
                        s.classList.remove('text-neutral-300');
                    } else {
                        s.classList.add('text-neutral-300');
                        s.classList.remove('text-secondary');
                    }
                });
            });
        });
        
        document.getElementById('rating').addEventListener('mouseout', () => {
            const activeStars = document.querySelectorAll('#rating i.text-secondary');
            if (activeStars.length === 0) {
                ratingStars.forEach(star => {
                    star.classList.add('text-neutral-300');
                    star.classList.remove('text-secondary');
                });
            }
        });
        
        // 评论筛选和排序功能
        const filterSelect = document.querySelector('select[value="all"]').parentElement.querySelector('select');
        const sortSelect = document.querySelector('select[value="newest"]').parentElement.querySelector('select');
        
        filterSelect.addEventListener('change', function() {
            // 这里可以添加筛选逻辑
            console.log('筛选条件变更为:', this.value);
        });
        
        sortSelect.addEventListener('change', function() {
            // 这里可以添加排序逻辑
            console.log('排序条件变更为:', this.value);
        });
        
        // 评论点赞功能
        const likeButtons = document.querySelectorAll('.fa-thumbs-up').forEach(button => {
            button.parentElement.addEventListener('click', function() {
                const countSpan = this.querySelector('span');
                let count = parseInt(countSpan.textContent.match(/\d+/)[0]);
                
                if (this.classList.contains('text-primary')) {
                    // 取消点赞
                    this.classList.remove('text-primary');
                    countSpan.textContent = `有用 (${count - 1})`;
                } else {
                    // 点赞
                    this.classList.add('text-primary');
                    countSpan.textContent = `有用 (${count + 1})`;
                }
            });
        });
    </script>
</body>
</html>
]
"""

# 使用lxml解析HTML
tree = etree.HTML(html_content)

# 定义XPath表达式
# 注意：以下XPath基于您提供的HTML结构，可能因实际页面变化而需要调整。
reviews_xpath = "//section[@class='space-y-6 mb-12']/article"
username_xpath = ".//div[1]/h3/text()"
dtime_xpath = ".//div[1]/p[@class='text-sm text-neutral-500']/text()"
score_full_star_xpath = ".//div[2]/div[@class='text-secondary mb-3']/*[name()='i' and contains(@class, 'fa-star') and not(contains(@class, 'fa-star-o')) and not(contains(@class, 'fa-star-half-o'))]"
score_half_star_xpath = ".//div[2]/div[@class='text-secondary mb-3']/*[name()='i' and contains(@class, 'fa-star-half-o')]"
title_xpath = ".//div[2]/h4/text()"
content_xpath = ".//div[2]/p[@class='text-neutral-700 mb-4']/text()"
img_xpath = ".//div[2]/div[contains(@class, 'flex gap-2')]/img"
surport_num_xpath = ".//div[contains(@class, 'flex items-center gap-6') or contains(@class, 'flex gap-6')]/button[1]/span/text()"
reply_num_xpath = ".//div[contains(@class, 'flex items-center gap-6') or contains(@class, 'flex gap-6')]/button[2]/span/text()"

# 存储所有评论数据的列表
all_reviews_data = []

# 遍历所有评论
reviews = tree.xpath(reviews_xpath)
if not reviews:
    print("未找到评论。请检查XPath表达式是否正确。")
else:
    for review in reviews:
        # 提取数据
        username = review.xpath(username_xpath)[0].strip() if review.xpath(username_xpath) else None
        dtime = review.xpath(dtime_xpath)[0].strip() if review.xpath(dtime_xpath) else None
        title = review.xpath(title_xpath)[0].strip() if review.xpath(title_xpath) else None
        content = review.xpath(content_xpath)[0].strip() if review.xpath(content_xpath) else None
        
        # 提取评分
        full_stars = len(review.xpath(score_full_star_xpath))
        half_stars = len(review.xpath(score_half_star_xpath))
        score = full_stars + half_stars * 0.5

        # 提取图片数量
        img_num = len(review.xpath(img_xpath))

        # 提取有用人数和回复数量，并处理文本
        surport_text = review.xpath(surport_num_xpath)[0].strip() if review.xpath(surport_num_xpath) else "0"
        surport_num_match = re.search(r'\d+', surport_text)
        surport_num = int(surport_num_match.group()) if surport_num_match else 0
        
        reply_text = review.xpath(reply_num_xpath)[0].strip() if review.xpath(reply_num_xpath) else "0"
        reply_num_match = re.search(r'\d+', reply_text)
        reply_num = int(reply_num_match.group()) if reply_num_match else 0

        # 将数据存入字典
        review_data = {
            "username评论人": username,
            "dtime评论时间": dtime,
            "score评分": score,
            "title评论标题": title,
            "content评论内容": content,
            "img_num图片数量": img_num,
            "surport_num有用人数": surport_num,
            "reply_num回复数量": reply_num
        }
        all_reviews_data.append(review_data)

# 打印结果
for data in all_reviews_data:
    print(data)
































