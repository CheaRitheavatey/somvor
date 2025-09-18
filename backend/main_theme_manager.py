class ThemeManager:
    def __init__(self):
        self.current_theme = "light"
        
        self.themes = {
            "light": {
                "bg": "#f0f0f0",
                "panel_bg": "#ffffff",
                "text": "#333333",
                "accent": "#4CAF50",
                "accent_hover": "#45a049",
                "error": "#f44336",
                "error_hover": "#d32f2f",
                "primary": "#2196F3",
                "secondary": "#FF9800",
                "border": "#e0e0e0"
            },
            "dark": {
                "bg": "#1a1a1a",
                "panel_bg": "#2d2d2d",
                "text": "#ffffff",
                "accent": "#4CAF50",
                "accent_hover": "#45a049",
                "error": "#f44336",
                "error_hover": "#d32f2f",
                "primary": "#2196F3",
                "secondary": "#FF9800",
                "border": "#404040"
            }
        }
        
    def set_theme(self, theme_name):
        """Set the current theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            
    def get_color(self, color_name):
        """Get a specific color from the current theme"""
        return self.themes[self.current_theme].get(color_name, "#000000")
        
    def get_all_colors(self):
        """Get all colors from the current theme"""
        return self.themes[self.current_theme]