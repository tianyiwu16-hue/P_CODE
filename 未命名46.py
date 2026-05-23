import pygame
import math
import random

# --- 核心配置 ---
WIDTH, HEIGHT = 1000, 700
BG_COLOR = (4, 4, 15)  # 深邃宇宙蓝背景
PARTICLE_NUM = 1500    # 粒子数量，越多越密（根据电脑性能调整）
HEART_SCALE = 14       # 爱心大小
SPEED = 1              # 动画速度

# 颜色盘：冰蓝、青色、白光
COLORS = [
    (0, 255, 255),    # 青色 (Cyan)
    (0, 150, 255),    # 天蓝 (Deep Sky Blue)
    (60, 200, 255),   # 亮蓝
    (200, 240, 255),  # 接近白色的蓝光
    (255, 255, 255)   # 纯白核心
]

# 图片里的文字 (可以自己修改)
TEXTS = ["李峋", "LOVE", "Python", "Forever", "Code"] 

# --- 初始化 ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cyberpunk Heart - Reconstruct")
font = pygame.font.SysFont("SimHei", 24, bold=True) # 使用支持中文的字体

# --- 核心类定义 ---

class Particle:
    def __init__(self, x, y, z):
        self.ox, self.oy, self.oz = x, y, z # 目标位置 (Origin)
        self.x, self.y, self.z = random.randint(-400, 400), random.randint(-300, 300), random.randint(-400, 400) # 当前位置 (初始随机)
        
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.vz = random.uniform(-2, 2)
        
        self.color = random.choice(COLORS)
        self.size = random.randint(1, 2)
        
        # 破碎参数
        self.scatter_state = False 
        self.return_speed = 0.04 # 回归速度

    def update(self, t, scatter):
        # 旋转公式 (绕Y轴)
        rad = t * 0.02
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        # 计算旋转后的目标点
        tx = self.ox * cos_r - self.oz * sin_r
        ty = self.oy
        tz = self.ox * sin_r + self.oz * cos_r
        
        # 破碎模式 (Scatter)
        if scatter:
            # 粒子受到噪点干扰，轻微乱飞
            self.x += math.sin(t * 5 + self.oy) * 2
            self.y += math.cos(t * 5 + self.ox) * 2
            # 慢慢远离中心
            self.x += (self.x - 0) * 0.01
            self.y += (self.y - 0) * 0.01
        else:
            # 重组模式：强力回归目标点
            self.x += (tx - self.x) * (self.return_speed + random.random() * 0.05)
            self.y += (ty - self.y) * (self.return_speed + random.random() * 0.05)
            self.z += (tz - self.z) * (self.return_speed + random.random() * 0.05)

    def draw(self, surface):
        # 3D 投影到 2D
        fov = 500
        dist = 5
        scale = fov / (self.z + fov + dist)
        
        sx = int(self.x * scale + WIDTH / 2)
        sy = int(self.y * scale + HEIGHT / 2)
        
        # 根据深度调整大小和亮度
        r_size = max(1, int(self.size * scale))
        
        # 简单的发光绘制
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            # 使用圆形绘制
            pygame.draw.circle(surface, self.color, (sx, sy), r_size)
            return (sx, sy)
        return None

class FloatingText:
    def __init__(self):
        self.text = random.choice(TEXTS)
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.alpha = 0
        self.fade_in = True
        self.color = random.choice(COLORS)
        
    def update(self):
        if self.fade_in:
            self.alpha += 2
            if self.alpha >= 255: self.fade_in = False
        else:
            self.alpha -= 2
            
        self.y -= 0.5 # 缓慢上浮

    def draw(self, surface):
        text_surf = font.render(self.text, True, self.color)
        text_surf.set_alpha(self.alpha)
        surface.blit(text_surf, (self.x, self.y))
        return self.alpha > 0

# --- 工具函数 ---
def get_heart_point():
    # 生成爱心坐标
    t = random.uniform(0, 2 * math.pi)
    u = random.uniform(0, math.pi)
    
    # 心形公式
    x = 16 * (math.sin(t) ** 3) * math.sin(u)
    y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) * math.sin(u)
    z = 6 * math.cos(u) * math.sin(t)
    
    return x * HEART_SCALE, y * HEART_SCALE, z * HEART_SCALE

# --- 主程序 ---
def main():
    clock = pygame.time.Clock()
    
    # 1. 生成粒子群
    particles = []
    for _ in range(PARTICLE_NUM):
        x, y, z = get_heart_point()
        # 加一点随机扩散，让它像星云一样蓬松
        x += random.gauss(0, 5)
        y += random.gauss(0, 5)
        z += random.gauss(0, 5)
        particles.append(Particle(x, y, z))
        
    floating_texts = []
    
    t = 0
    running = True
    scatter_mode = False # 破碎模式开关
    
    # 创建一个支持透明和发光的图层
    glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    while running:
        screen.fill(BG_COLOR)
        glow_surface.fill((0,0,0,0)) # 清空透明层
        
        t += SPEED
        
        # 自动触发“破碎重组”循环
        # 每300帧破碎一次，持续50帧
        if t % 350 > 300: 
            scatter_mode = True
        else:
            scatter_mode = False
            
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN: # 点击鼠标手动破碎
                scatter_mode = not scatter_mode

        # 绘制所有粒子
        coords_2d = []
        for p in particles:
            p.update(t, scatter_mode)
            pos = p.draw(glow_surface)
            if pos:
                coords_2d.append(pos)
                
        # 绘制“数据链路”连线 (视觉核心)
        # 为了不卡顿，我们只随机连接部分粒子
        if not scatter_mode: # 破碎时断开连接
            for _ in range(30): # 每帧随机画30条线
                p1 = random.choice(coords_2d)
                p2 = random.choice(coords_2d)
                # 计算距离，只有近的才连线
                dist = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                if dist < 4000: # 距离阈值
                    # 线条越近越亮
                    alpha = max(0, 100 - int(dist/40))
                    pygame.draw.line(glow_surface, (0, 255, 255, alpha), p1, p2, 1)

        # 绘制悬浮文字
        if random.random() < 0.02: # 随机生成新文字
            floating_texts.append(FloatingText())
        
        # 更新文字并移除消失的
        floating_texts = [txt for txt in floating_texts if txt.draw(screen)]

        # 合成画面：使用 BLEND_ADD 叠加发光层
        screen.blit(glow_surface, (0, 0), special_flags=pygame.BLEND_ADD)
        
        # 底部状态栏
        status = "SYSTEM: RECONSTRUCTING..." if not scatter_mode else "SYSTEM: CRITICAL ERROR - SCATTERING"
        info = font.render(status, True, (0, 255, 255))
        screen.blit(info, (20, HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
    
    
    
    
    
    



