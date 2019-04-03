DISPLAY_SIZE = 6
class CompareItemBase():
    def __init__(self, path, display_name, surfix=''):
        self.path = path
        self.display_name = display_name
        self.surfix = surfix
def hide_ax_frame(ax):
        ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        ax.set_xticklabels([])
        plt.sca(ax)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        
def show_compare_images(original_file_paths, compare_item_bases, original_display_name='origin'
                                                 , hide_frame=True, padding_item=0, row_data_count=1, with_data_number=True):
    if len(original_file_paths) < 1:
        print('compare data is Nothing .')
        return
    if len(compare_item_bases) < 1:
        print('compare path is Nothing .')
    
    compare_item_bases.insert(0, CompareItemBase(os.path.dirname(original_file_paths[0]), original_display_name))
    for k, origin_path in enumerate(original_file_paths):
        print(os.path.basename(origin_path))
        base_name, ext = os.path.splitext(os.path.basename(origin_path))
        compare_item_count = len(compare_item_bases) + padding_item
        if k % row_data_count == 0:
            fig = plt.figure(k, figsize=(DISPLAY_SIZE * compare_item_count * row_data_count, DISPLAY_SIZE))
        axes = []
        for j, cbi in enumerate(compare_item_bases):
            ax = fig.add_subplot(1, compare_item_count * row_data_count, j + 1 + (k % row_data_count * compare_item_count))
            if with_data_number:
                if cbi.display_name == '':
                    plt.title('[%d]' % k)
                else:
                    plt.title('[%d] : %s' % (k, cbi.display_name))
            else:
                plt.title(cbi.display_name)
            if hide_frame:
                hide_ax_frame(ax)
            axes.append(ax)
        for i, (ax, cbi) in enumerate(zip(axes, compare_item_bases)):
            item_path = os.path.join(cbi.path, base_name + cbi.surfix + ext)
            if os.path.exists(item_path):
                img = cv2.cvtColor(cv2.imread(item_path), cv2.COLOR_BGR2RGB)
                fig.sca(ax)
                plt.imshow(img)
            else:
                print('not exists : ', item_path)
        if (k + 1) % row_data_count == 0:
            plt.show()
def show_img_list(files, col_item_count=5, disp_size=3):
    for k, file in enumerate(files):
        c = k + 1
        ax_id = c % col_item_count
        if ax_id == 1:
            fig = plt.figure(figsize=(disp_size * col_item_count, disp_size))
        if ax_id == 0:
            ax_id = col_item_count
        ax = fig.add_subplot(1, col_item_count, ax_id)
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(os.path.basename(file))
        hide_ax_frame(ax)
        if ax_id == col_item_count:
            plt.show()
            
class ImageLinePlotter():
    def __init__(self, fig_id, plot_area_num=6, display_size=DISPLAY_SIZE):
        self.fig_id = fig_id
        self.display_size = display_size
        self.plot_area_num = plot_area_num
        self.figsize = (self.plot_area_num*self.display_size, self.display_size)
        self.fig = plt.figure(self.fig_id, figsize=self.figsize)
        self.image_infos = [None] * self.plot_area_num
        self.image_count = 0
        
    def add_image(self, img, title='', pos=None):
        if pos is None:
            pos = len(self.image_infos) - 1
            for k, img_info in enumerate(self.image_infos):
                if img_info is None:
                    pos = k+1
                    break
        self.image_infos[pos-1] = (img, title)
    def show_plot(self):
        for k, img_info in enumerate(self.image_infos):
            if img_info is None:
                continue
            (img, title) = img_info
            ax = self.fig.add_subplot(1, self.plot_area_num, k + 1)
            plt.title(title)
            hide_ax_frame(ax)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()