import pandas as pd
pd.set_option('precision', 16)


class convert_json_to_md():
    def __init__(self, json_path, save_path=None):
        self.json_path = json_path
        self.json_object = pd.read_json(json_path)
        if save_path is None:
            self.save_path = json_path[:-5] + '.md'
        else:
            self.save_path = save_path

    def to_markdown(self):
        return self.json_object.to_markdown(self.save_path)

    def to_markdown_string(self):
        return self.json_object.to_markdown()


if __name__ == '__main__':
    convert = convert_json_to_md(json_path='./json_markdown.json')
    convert.to_markdown()
    print(convert.to_markdown_string())